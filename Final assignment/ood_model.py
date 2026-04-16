import torch
import torch.nn as nn
import normflows as nf


class OOD_Detector(nn.Module):
	"""Normalizing-flow based OOD detector operating on DINOv3 token features.

	Expected token input shape is ``[B, T, D]`` where:
	- ``B``: batch size
	- ``T``: number of tokens (patch tokens, or patch+CLS if desired)
	- ``D``: token embedding size from DINOv3 backbone
	"""

	def __init__(
		self,
		token_dim: int,
		flow_dim: int = 128,
		hidden_dim: int = 256,
		num_flow_layers: int = 8,
		token_sample_size: int | None = 4096,
	):
		super().__init__()
		if token_dim <= 0:
			raise ValueError("token_dim must be > 0")
		if flow_dim <= 1:
			raise ValueError("flow_dim must be > 1")
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be > 0")
		if num_flow_layers <= 0:
			raise ValueError("num_flow_layers must be > 0")

		self.token_dim = token_dim
		self.flow_dim = flow_dim
		self.hidden_dim = hidden_dim
		self.num_flow_layers = num_flow_layers
		self.token_sample_size = token_sample_size

		# Lightweight projection from high-dimensional DINO tokens to flow space.
		self.token_projector = nn.Sequential(
			nn.LayerNorm(self.token_dim),
			nn.Linear(self.token_dim, self.hidden_dim),
			nn.GELU(),
			nn.Linear(self.hidden_dim, self.flow_dim),
		)

		self.nf_model = self._build_flow()
		self.threshold: float | None = None
		self.id_score_mean: float | None = None
		self.id_score_std: float | None = None

	def _build_flow(self) -> nf.NormalizingFlow:
		base = nf.distributions.base.DiagGaussian(self.flow_dim) # Simple isotropic Gaussian base distribution in flow space (latent space)
		flows = []

		for layer_idx in range(self.num_flow_layers):
			mask = self._alternating_mask(self.flow_dim, invert=(layer_idx % 2 == 1))
			t_net = nf.nets.MLP(
				[self.flow_dim, self.hidden_dim, self.hidden_dim, self.flow_dim],
				init_zeros=True,
			)
			s_net = nf.nets.MLP(
				[self.flow_dim, self.hidden_dim, self.hidden_dim, self.flow_dim],
				init_zeros=True,
				output_fn="tanh",
				output_scale=2.0,
			)
			flows.append(nf.flows.MaskedAffineFlow(mask, t=t_net, s=s_net))

		return nf.NormalizingFlow(base, flows)

	@staticmethod
	def _alternating_mask(dim: int, invert: bool = False) -> torch.Tensor:
		mask = torch.arange(dim) % 2
		if invert:
			mask = 1 - mask
		return mask.float()

	def _check_tokens(self, tokens: torch.Tensor) -> None:
		if tokens.dim() != 3:
			raise ValueError(
				f"Expected tokens with shape [B, T, D], got tensor with shape {tuple(tokens.shape)}"
			)
		if tokens.size(-1) != self.token_dim:
			raise ValueError(
				f"Expected token_dim={self.token_dim}, but got D={tokens.size(-1)}"
			)

	def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
		self._check_tokens(tokens)
		flat_tokens = tokens.reshape(-1, self.token_dim)
		return self.token_projector(flat_tokens)

	def _sample_tokens(self, projected_tokens: torch.Tensor) -> torch.Tensor:
		if self.token_sample_size is None or projected_tokens.size(0) <= self.token_sample_size:
			return projected_tokens

		indices = torch.randperm(projected_tokens.size(0), device=projected_tokens.device)[
			: self.token_sample_size
		]
		return projected_tokens[indices]

	def forward(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Return OOD score per image (higher means more likely OOD)."""
		token_scores = self.score_tokens(tokens)
		return token_scores.mean(dim=1)

	def loss(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Negative log-likelihood loss for training on in-distribution tokens."""
		projected_tokens = self._project_tokens(tokens)
		sampled_tokens = self._sample_tokens(projected_tokens)
		return self.nf_model.forward_kld(sampled_tokens)

	def score_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
		"""Return token-level OOD scores of shape [B, T]."""
		self._check_tokens(tokens)
		bsz, n_tokens, _ = tokens.shape
		projected_tokens = self._project_tokens(tokens)
		log_prob = self.nf_model.log_prob(projected_tokens)
		ood_score = -log_prob
		return ood_score.view(bsz, n_tokens)

	@torch.no_grad()
	def calibrate_threshold(self, id_tokens: torch.Tensor, quantile: float = 0.95) -> float:
		"""Set threshold from ID data using image-level OOD score quantile."""
		if not 0.0 < quantile < 1.0:
			raise ValueError("quantile must be in (0, 1)")

		scores = self.forward(id_tokens)
		self.id_score_mean = float(scores.mean().item())
		self.id_score_std = float(scores.std(unbiased=False).clamp_min(1e-6).item())
		threshold = torch.quantile(scores, quantile).item()
		self.threshold = float(threshold)
		return self.threshold

	def set_score_calibration(self, mean: float, std: float) -> None:
		"""Set score distribution statistics used for probability conversion."""
		if std <= 0:
			raise ValueError("std must be > 0")
		self.id_score_mean = float(mean)
		self.id_score_std = float(std)

	def score_to_probability(self, scores: torch.Tensor) -> torch.Tensor:
		"""Map raw OOD scores to [0, 1] using calibrated ID score statistics.

		Interpretation: approximately the ID-score percentile under a Gaussian fit.
		Higher values mean more likely OOD.
		"""
		z = (scores - self.id_score_mean) / max(self.id_score_std, 1e-6)
		return 0.5 * (1.0 + torch.erf(z / 1.4142135623730951))

	@torch.no_grad()
	def predict_ood(
		self,
		tokens: torch.Tensor,
		threshold: float | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Return OOD boolean mask and image-level OOD scores.

		Outputs:
		- ``included`` with shape ``[B]``
		- ``scores`` with shape ``[B]`` (raw scores or probabilities)
		"""
		use_threshold = self.threshold if threshold is None else threshold
		if use_threshold is None:
			raise ValueError("No threshold provided. Call calibrate_threshold or pass threshold.")

		raw_scores = self.forward(tokens)
		prob = self.score_to_probability(raw_scores)
		included = prob < use_threshold # Note: lower probability means more likely ID, so we include the image if prob < threshold
		return included, prob


class OOD_Detector_v2(OOD_Detector):
	"""OOD detector v2 with neural spline coupling flows.

	This class keeps the same constructor and public API as ``OOD_Detector``
	so it can be swapped in without changing call sites.
	"""

	def _build_flow(self) -> nf.NormalizingFlow:
		base = nf.distributions.base.DiagGaussian(self.flow_dim)
		flows = []

		for layer_idx in range(self.num_flow_layers):
			flows.append(
				nf.flows.CoupledRationalQuadraticSpline(
					num_input_channels=self.flow_dim,
					num_blocks=2,
					num_hidden_channels=self.hidden_dim,
					num_bins=8,
					tails="linear",
					tail_bound=3.0,
					reverse_mask=(layer_idx % 2 == 1),
				)
			)
			flows.append(nf.flows.Permute(self.flow_dim, mode="swap"))

		return nf.NormalizingFlow(base, flows)


