"""Coarse-to-fine search of a dictionary simulated on a lattice of parameter values."""

from __future__ import annotations

__all__ = ["Lattice"]

import itertools
from dataclasses import dataclass, replace

import torch

from ._grouped import best_of, correlate


@dataclass(frozen=True)
class Lattice:
    """Where each atom sits on the lattice its parameters were simulated on.

    A dictionary simulated on a product of value lists, one per parameter,
    possibly with some combinations left out (T2 above T1, for instance), is
    searched in two passes: every ``stride``-th value along each axis, then
    the ``2 stride + 1`` values around the winner along every axis. With a
    smooth dependence of the signal on the parameters the winner of the
    coarse pass lies within a stride of the best atom, and the search costs
    ``atoms / stride^P + (2 stride + 1)^P`` inner products per voxel instead
    of ``atoms``.

    Attributes
    ----------
    positions : torch.Tensor
        ``(atoms, P)`` index of each atom's value along each axis.
    table : torch.Tensor
        The lattice, ``P`` axes of ``int64`` atom indices; ``-1`` where no atom
        was simulated.
    coarse : torch.Tensor
        Indices of the atoms on every ``stride``-th value of every axis.
    offsets : torch.Tensor
        ``(neighbours, P)`` steps from the coarse winner the second pass visits.
    """

    positions: torch.Tensor
    table: torch.Tensor
    coarse: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def fit(cls, parameters: torch.Tensor, stride: int) -> Lattice:
        """Place ``(atoms, P)`` parameter values on their lattice.

        Raises
        ------
        ValueError
            If two atoms share every parameter value, so the lattice cannot
            tell them apart, or ``stride`` is below one.
        """
        if stride < 1:
            raise ValueError(f"the coarse stride is at least one, got {stride}")
        parameters = torch.as_tensor(parameters)
        if parameters.ndim != 2 or parameters.shape[1] < 1:
            raise ValueError("coarse-to-fine search needs parameters (atoms, P)")
        columns = [torch.unique(c, return_inverse=True) for c in parameters.mT]
        positions = torch.stack([where for _, where in columns], -1)
        extent = [len(values) for values, _ in columns]
        table = torch.full(extent, -1, dtype=torch.int64, device=parameters.device)
        table[tuple(positions.mT)] = torch.arange(
            parameters.shape[0], device=parameters.device
        )
        if int((table >= 0).sum()) != parameters.shape[0]:
            raise ValueError(
                "two atoms have the same parameter values, so the lattice cannot "
                "tell them apart"
            )
        coarse = (positions % stride == 0).all(-1).nonzero()[:, 0]
        steps = range(-stride, stride + 1)
        offsets = torch.tensor(
            list(itertools.product(steps, repeat=len(extent))),
            dtype=torch.int64,
            device=parameters.device,
        )
        return cls(positions=positions, table=table, coarse=coarse, offsets=offsets)

    def to(self, device: torch.device | str) -> Lattice:
        """The same lattice with its tensors on ``device``."""
        return replace(
            self,
            positions=self.positions.to(device),
            table=self.table.to(device),
            coarse=self.coarse.to(device),
            offsets=self.offsets.to(device),
        )

    def search(
        self,
        signals: torch.Tensor,
        atoms: torch.Tensor,
        top_k: int,
        refine: bool,
        chunk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The best ``top_k`` atoms of each normalized signal, and their scores.

        Parameters
        ----------
        signals : torch.Tensor
            ``(voxels, contrasts)``, normalized.
        atoms : torch.Tensor
            ``(atoms, contrasts)``, normalized, in the order of :attr:`positions`.
        top_k : int
            Candidates kept.
        refine : bool
            Search the neighbourhood of the coarse winner; ``False`` returns
            the coarse pass alone.
        chunk : int
            Voxels scored at once.
        """
        indices, scores = [], []
        candidates = atoms[self.coarse]
        upper = torch.tensor(self.table.shape, device=self.table.device) - 1
        for part in signals.split(chunk):
            best_scores, best = best_of(
                correlate(part, candidates), min(top_k, candidates.shape[0])
            )
            best = self.coarse[best]
            if refine:
                centre = self.positions[best[:, 0]]
                around = centre[:, None, :] + self.offsets[None]
                inside = ((around >= 0) & (around <= upper)).all(-1)
                around = around.clamp(torch.zeros_like(upper), upper)
                near = self.table[tuple(around.permute(2, 0, 1))]
                near = near.masked_fill(~inside, -1)
                dtype = torch.promote_types(part.dtype, atoms.dtype)
                found = torch.einsum(
                    "nc,nac->na",
                    part.to(dtype),
                    atoms[near.clamp_min(0)].to(dtype).conj(),
                ).abs()
                found = found.masked_fill(near < 0, -1.0)
                keep = min(top_k, near.shape[1])
                best_scores, choice = best_of(found, keep)
                best = torch.gather(near, -1, choice)
            indices.append(best)
            scores.append(best_scores)
        return torch.cat(indices), torch.cat(scores)
