"""Minimal XLB + Carbon + Warp-fork import/smoke check (run inside the CUDA-12 container)."""

import warp as wp

import xlb


def main() -> None:
    repo = getattr(wp, "__file__", "") or ""
    assert "carbon/warp" in repo.replace("\\", "/") or "nvlukasz" in repo, (
        f"expected the Carbon Warp fork on the path, got {repo!r}"
    )
    print("warp fork:", repo)
    print("xlb:", getattr(xlb, "__file__", "?"))

    import carbon

    gate = carbon.init()
    print("carbon device_count:", gate.lib.carbonpy_device_count())
    print("XLB + Carbon + Warp-fork import OK")


if __name__ == "__main__":
    main()
