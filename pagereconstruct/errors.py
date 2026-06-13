"""Errors for the PAGERECONSTRUCT unit."""

from __future__ import annotations


class PageReconstructError(Exception):
    pass


class PageReconstructInputError(PageReconstructError):
    pass


class PageRenderPlanError(PageReconstructError):
    pass


class PageRenderBackendError(PageReconstructError):
    pass
