"""aggregate() resume correctamente una lista de FrameProfile."""
from studies.segmentation.core.profiling import FrameProfile, CropTiming
from studies.segmentation.core.timing_stats import aggregate


def _profile(total: float, enc: float, dec: float, n: int) -> FrameProfile:
    return FrameProfile(total_ms=total, n_masks=n, crops=[CropTiming(encoder_ms=enc, decode_ms=dec, peak_vram_mb=100.0)])


def test_aggregate_mean_and_extremes():
    profs = [_profile(100, 20, 60, 10), _profile(200, 40, 120, 20)]
    ts = aggregate(profs)
    assert ts.n == 2
    assert ts.total_ms.mean == 150 and ts.total_ms.min == 100 and ts.total_ms.max == 200
    assert ts.encoder_ms.mean == 30
    assert ts.decode_ms.mean == 90
    assert ts.n_masks.mean == 15
    assert ts.n_crops == 1


def test_frameprofile_derived_sums():
    p = FrameProfile(total_ms=300, n_masks=5, crops=[
        CropTiming(encoder_ms=10, decode_ms=50, peak_vram_mb=200),
        CropTiming(encoder_ms=15, decode_ms=70, peak_vram_mb=300),
    ])
    assert p.n_crops == 2
    assert p.encoder_ms == 25
    assert p.decode_ms == 120
    assert p.peak_vram_mb == 300
