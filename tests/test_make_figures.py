from tools import make_figures


def test_presentation_figures_are_generated(tmp_path):
    files = make_figures.main(["--out", str(tmp_path)])
    assert [f.name for f in files] == ["fig1_area_weighting.png", "fig2_coverage.png", "fig3_graycard.png",
                                      "fig4_reject_curve_demo.png", "fig5_shadow.png", "fig6_top_vs_multiview.png"]
    assert all(f.stat().st_size > 10_000 for f in files)
