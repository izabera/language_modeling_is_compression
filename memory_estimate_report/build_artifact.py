"""Build the reproducible memory-estimate dataset and portable report input."""

from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path


GENERATED_AT = "2026-07-18T23:25:42Z"
GIB = 1 << 30
BATCH_SIZES = (1, 4, 16, 32, 64, 128)
REPORT_DIR = Path(__file__).resolve().parent

PARAMETER_COUNTS = {
    "200k": 231_936,
    "800k": 856_832,
    "3.2m": 3_286_272,
    "6.4m": 6_441_216,
    "38m": 38_066_432,
}

# (argument bytes, output bytes, temporary bytes), returned by XLA's
# CompiledMemoryStats for the current blockwise implementation.
CURRENT = {
    "200k": {
        1: (2_785_284, 2_784_492, 63_987_608),
        4: (2_791_428, 2_784_492, 270_803_608),
        16: (2_816_004, 2_784_492, 1_085_414_488),
        32: (2_848_772, 2_784_492, 2_170_821_720),
        64: (2_914_308, 2_784_492, 4_341_636_184),
        128: (3_045_380, 2_784_492, 8_683_265_112),
    },
    "800k": {
        1: (10_284_036, 10_283_244, 121_667_736),
        4: (10_290_180, 10_283_244, 515_909_272),
        16: (10_314_756, 10_283_244, 2_064_531_096),
        32: (10_347_524, 10_283_244, 4_129_561_688),
        64: (10_413_060, 10_283_244, 8_267_504_728),
        128: (10_544_132, 10_283_244, 16_535_002_200),
    },
    "3.2m": {
        1: (39_437_316, 39_436_524, 237_011_608),
        4: (39_443_460, 39_436_524, 1_006_644_888),
        16: (39_468_036, 39_436_524, 4_026_556_056),
        32: (39_500_804, 39_436_524, 8_053_104_280),
        64: (39_566_340, 39_436_524, 16_110_919_320),
        128: (39_697_412, 39_436_524, 32_221_699_160),
    },
    "6.4m": {
        1: (77_296_644, 77_297_004, 463_511_192),
        4: (77_302_788, 77_297_004, 1_979_730_584),
        16: (77_327_364, 77_297_004, 7_918_877_336),
        32: (77_360_132, 77_297_004, 15_837_739_672),
        64: (77_425_668, 77_297_004, 31_680_707_224),
        128: (77_556_740, 77_297_004, 63_361_267_800),
    },
    "38m": {
        1: (456_799_236, 456_800_748, 1_075_887_768),
        4: (456_805_380, 456_800_748, 5_888_865_944),
        16: (456_829_956, 456_800_748, 23_555_266_200),
        32: (456_862_724, 456_800_748, 47_110_510_232),
        64: (456_928_260, 456_800_748, 94_220_998_296),
        128: (457_059_332, 456_800_748, 188_441_974_424),
    },
}

# The same fields for the previous dense-attention implementation. These are
# exact compiler measurements at B=1, 4 and 16. Larger batches are estimated
# below from the B=4-to-B=16 affine slope.
PREVIOUS_MEASURED = {
    "200k": {
        1: (2_785_284, 2_784_492, 1_788_887_044),
        4: (2_791_428, 2_784_492, 6_689_390_596),
        16: (2_816_004, 2_784_492, 26_759_790_596),
    },
    "800k": {
        1: (10_284_036, 10_283_244, 1_830_821_892),
        4: (10_290_180, 10_283_244, 6_908_542_980),
        16: (10_314_756, 10_283_244, 27_638_235_140),
    },
    "3.2m": {
        1: (39_437_316, 39_436_524, 1_914_707_972),
        4: (39_443_460, 39_436_524, 7_348_420_612),
        16: (39_468_036, 39_436_524, 29_395_779_588),
    },
    "6.4m": {
        1: (77_296_644, 77_297_004, 3_726_647_300),
        4: (77_302_788, 77_297_004, 14_663_286_788),
        16: (77_327_364, 77_297_004, 58_655_244_292),
    },
    "38m": {
        1: (456_799_236, 456_800_748, 5_830_098_948),
        4: (456_805_380, 456_800_748, 24_612_208_644),
        16: (456_829_956, 456_800_748, 98_448_703_492),
    },
}


def total_bytes(fields: tuple[int, int, int]) -> int:
    # alias_size_in_bytes was zero for every compilation.
    return sum(fields)


def previous_fields(preset: str, batch_size: int) -> tuple[int, int, int, bool]:
    measured = PREVIOUS_MEASURED[preset]
    if batch_size in measured:
        return (*measured[batch_size], True)

    total4 = total_bytes(measured[4])
    total16 = total_bytes(measured[16])
    slope = Fraction(total16 - total4, 12)
    intercept = Fraction(total4) - 4 * slope
    estimated_total = round(intercept + batch_size * slope)
    argument = measured[1][0] + (batch_size - 1) * 2_048
    output = measured[1][1]
    temporary = estimated_total - argument - output
    return argument, output, temporary, False


def display_gib(value: float) -> str:
    return f"{value:.3f}" if value < 1 else f"{value:.2f}"


def build_rows() -> list[dict[str, object]]:
    rows = []
    for preset, parameter_count in PARAMETER_COUNTS.items():
        for batch_size in BATCH_SIZES:
            current_argument, current_output, current_temporary = CURRENT[preset][batch_size]
            current_total = current_argument + current_output + current_temporary
            previous_argument, previous_output, previous_temporary, previous_exact = previous_fields(
                preset, batch_size
            )
            previous_total = previous_argument + previous_output + previous_temporary
            rows.append(
                {
                    "preset": preset,
                    "parameter_count": parameter_count,
                    "batch_size": batch_size,
                    "sequence_length": 2_048,
                    "previous_argument_bytes": previous_argument,
                    "previous_output_bytes": previous_output,
                    "previous_temporary_bytes": previous_temporary,
                    "previous_total_bytes": previous_total,
                    "previous_total_gib": previous_total / GIB,
                    "previous_measurement": "compiled" if previous_exact else "affine estimate",
                    "current_argument_bytes": current_argument,
                    "current_output_bytes": current_output,
                    "current_temporary_bytes": current_temporary,
                    "current_total_bytes": current_total,
                    "current_total_gib": current_total / GIB,
                    "current_measurement": "compiled",
                    "reduction_gib": (previous_total - current_total) / GIB,
                    "reduction_ratio": previous_total / current_total,
                    "suggested_available_ram_gib": 1.25 * current_total / GIB + 2,
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    output = REPORT_DIR / "memory_estimates.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact(rows: list[dict[str, object]]) -> dict[str, object]:
    lookup = {(row["preset"], row["batch_size"]): row for row in rows}
    matrix = []
    for preset, parameter_count in PARAMETER_COUNTS.items():
        matrix_row: dict[str, object] = {
            "preset": preset,
            "parameter_count": parameter_count,
        }
        for batch_size in BATCH_SIZES:
            row = lookup[(preset, batch_size)]
            marker = "≈" if row["previous_measurement"] != "compiled" else ""
            matrix_row[f"b{batch_size}"] = (
                f"{marker}{display_gib(float(row['previous_total_gib']))} → "
                f"{display_gib(float(row['current_total_gib']))}"
            )
        matrix.append(matrix_row)

    batch1_savings = []
    for preset, parameter_count in PARAMETER_COUNTS.items():
        row = lookup[(preset, 1)]
        batch1_savings.append(
            {
                "preset": preset,
                "parameter_count": parameter_count,
                "reduction_ratio": round(float(row["reduction_ratio"]), 3),
                "previous_gib": round(float(row["previous_total_gib"]), 3),
                "current_gib": round(float(row["current_total_gib"]), 3),
            }
        )

    current_batch128 = []
    for preset, parameter_count in PARAMETER_COUNTS.items():
        row = lookup[(preset, 128)]
        current_batch128.append(
            {
                "preset": preset,
                "parameter_count": parameter_count,
                "current_gib": round(float(row["current_total_gib"]), 3),
                "suggested_available_ram_gib": round(
                    float(row["suggested_available_ram_gib"]), 1
                ),
            }
        )

    compiled_source = {
        "id": "compiled_memory_measurements",
        "label": "JAX/XLA CPU compiled-memory measurements",
        "path": "memory_estimate_report/memory_estimates.csv",
        "query": {
            "engine": "JAX/XLA CPU 0.10.2",
            "language": "sql",
            "sql": (
                "SELECT preset, parameter_count, batch_size, previous_total_gib, "
                "previous_measurement, current_total_gib, reduction_ratio\n"
                "FROM read_csv_auto('memory_estimate_report/memory_estimates.csv')\n"
                "ORDER BY parameter_count, batch_size"
            ),
            "description": (
                "Compile-only estimates for uint8 [batch, 2048] inputs, FP32 model state, "
                "and Adam. Current values are compiled at every batch size. Previous values "
                "are compiled at batches 1, 4, and 16; batches 32-128 use the validated "
                "affine batch scaling from batches 4-16."
            ),
            "executed_at": GENERATED_AT,
        },
    }
    implementation_source = {
        "id": "model_implementation",
        "label": "Transformer implementation and presets",
        "path": "transformer.py",
        "query": {
            "engine": "local source",
            "language": "sql",
            "sql": "SELECT content FROM read_text('transformer.py')",
            "description": (
                "Defines the five model presets, FP32 defaults, causal self-attention, and "
                "the current 128-token query/key block size."
            ),
            "executed_at": GENERATED_AT,
        },
    }

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Transformer training-memory estimate",
            "description": (
                "Compiler-estimated peak live buffers before and after blockwise attention, "
                "for every model preset and requested batch size."
            ),
            "generatedAt": GENERATED_AT,
            "cards": [],
            "charts": [
                {
                    "id": "batch1_reduction",
                    "title": "Batch-1 memory reduction by preset",
                    "subtitle": (
                        "Ratio of previous dense-attention footprint to current blockwise footprint; "
                        "higher is better."
                    ),
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "batch1_savings",
                    "sourceId": "compiled_memory_measurements",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "preset", "type": "nominal", "label": "Preset"},
                        "y": {
                            "field": "reduction_ratio",
                            "type": "quantitative",
                            "label": "Memory reduction (×)",
                            "format": "number",
                        },
                        "tooltip": [
                            {"field": "preset", "type": "nominal", "label": "Preset"},
                            {
                                "field": "reduction_ratio",
                                "type": "quantitative",
                                "label": "Reduction (×)",
                                "format": "number",
                            },
                            {
                                "field": "previous_gib",
                                "type": "quantitative",
                                "label": "Previous (GiB)",
                                "format": "number",
                            },
                            {
                                "field": "current_gib",
                                "type": "quantitative",
                                "label": "Current (GiB)",
                                "format": "number",
                            },
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "memory_matrix",
                    "title": "Previous dense → current blockwise memory",
                    "subtitle": (
                        "GiB per full FP32 Adam update at sequence length 2048. "
                        "≈ marks extrapolated previous values; all current values are compiled."
                    ),
                    "showDescription": True,
                    "dataset": "comparison_matrix",
                    "sourceId": "compiled_memory_measurements",
                    "columns": [
                        {"field": "preset", "label": "Preset", "type": "text"},
                        {
                            "field": "parameter_count",
                            "label": "Actual parameters",
                            "type": "number",
                            "format": "number",
                        },
                        {"field": "b1", "label": "Batch 1", "type": "text"},
                        {"field": "b4", "label": "Batch 4", "type": "text"},
                        {"field": "b16", "label": "Batch 16", "type": "text"},
                        {"field": "b32", "label": "Batch 32", "type": "text"},
                        {"field": "b64", "label": "Batch 64", "type": "text"},
                        {"field": "b128", "label": "Batch 128", "type": "text"},
                    ],
                    "defaultSort": {"field": "parameter_count", "direction": "asc"},
                }
            ],
            "sources": [
                {key: value for key, value in compiled_source.items() if key != "query"},
                {key: value for key, value in implementation_source.items() if key != "query"},
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "layout": "full",
                    "body": "# Transformer training-memory estimate",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "compiled_memory_measurements",
                    "body": (
                        "## Technical summary\n\n"
                        "Blockwise attention materially changes the fit envelope. At batch 1, "
                        "the compiler estimate falls from **1.671 to 0.065 GiB (25.8×)** for the "
                        "200k preset and from **6.281 to 1.853 GiB (3.4×)** for the 38m preset. "
                        "The gain is largest in small models, where the old 2048×2048 attention "
                        "matrices dominated nearly everything else.\n\n"
                        "The improvement does not make large batches cheap: the current 38m preset "
                        "is **22.79 GiB at batch 16** and **176.35 GiB at batch 128** before process "
                        "overhead."
                    ),
                },
                {
                    "id": "comparison_intro",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Previous versus current footprint\n\n"
                        "Each cell is **previous dense → current blockwise**. GiB means 2³⁰ bytes. "
                        "Previous batches 1, 4, and 16 and every current value are direct compiler "
                        "estimates; the ≈ values are linear extrapolations for the old implementation."
                    ),
                },
                {
                    "id": "comparison_table",
                    "type": "table",
                    "layout": "full",
                    "tableId": "memory_matrix",
                },
                {
                    "id": "savings_intro",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Where blockwise attention helps most\n\n"
                        "At batch 1 the reduction ranges from 25.8× for 200k to 3.4× for 38m. "
                        "As the model grows, parameter, optimizer-state, and feed-forward memory "
                        "occupy more of the total, so removing the quadratic attention materialization "
                        "has a smaller—but still substantial—relative effect."
                    ),
                },
                {
                    "id": "savings_chart",
                    "type": "chart",
                    "layout": "full",
                    "chartId": "batch1_reduction",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Scope and metric definition\n\n"
                        "This is a **training-step buffer estimate**, not whole-process RSS. The "
                        "compiled target is `_update_parameters` with a uint8 `[batch, 2048]` batch, "
                        "vocabulary 256, FP32 parameters/activations, Adam state, no gradient clipping, "
                        "no donated arguments, and the current 128-token attention blocks.\n\n"
                        "The reported total is `argument + output + temporary − alias` bytes. Alias "
                        "was zero in every current compilation. Because arguments are not donated, "
                        "both incoming and returned parameter/Adam trees can be live in the estimate."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Methodology\n\n"
                        "All 30 current configurations were lowered and compiled on the CPU backend "
                        "and read through `CompiledMemoryStats`; they were not executed. The previous "
                        "dense-attention code was compiled at batches 1, 4, and 16. For batches 32, "
                        "64, and 128, the old total uses the affine slope between batches 4 and 16. "
                        "A batch-8 spot check missed that fit by less than 0.001 GiB."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Limitations and robustness\n\n"
                        "XLA memory analysis is backend- and version-specific and is intended as an "
                        "estimate. It excludes Python/JAX runtime RSS, compiler working memory, allocator "
                        "caching and fragmentation, executable/constants that the report does not expose, "
                        "host copies, OS/library overhead, and the roughly 100 MB eager dataset. The old "
                        "Python attention path also constructed large float64 causal masks while tracing; "
                        "that tracing spike is not added to the table because XLA can fold or broadcast "
                        "those constants differently. Actual peak RSS should therefore be measured before "
                        "setting a hard machine-size limit."
                    ),
                },
                {
                    "id": "recommendations",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Practical sizing and next steps\n\n"
                        "For an initial CPU capacity check, budget roughly **1.25× the current table "
                        "+ 2 GiB** of available RAM, then validate with process-level peak RSS. On a "
                        "32 GiB machine this suggests 200k/800k through batch 128, 3.2m through batch "
                        "64, 6.4m through batch 32, and 38m at batch 4 comfortably; 38m batch 16 is "
                        "possible by the buffer estimate but leaves little compiler/runtime headroom.\n\n"
                        "The next low-risk memory experiment is argument donation for the parameter and "
                        "optimizer-state trees. It could remove up to one incoming state copy—about 436 "
                        "MiB for 38m—but requires checking call-site ownership and recompilation behavior."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## Further questions\n\n"
                        "The useful follow-up is to record peak RSS for one small and one large preset, "
                        "including first-step compilation and steady-state steps. That will calibrate the "
                        "headroom rule and show whether compiler memory, buffer memory, or dataset/runtime "
                        "overhead is now the practical constraint."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "comparison_matrix": matrix,
                "batch1_savings": batch1_savings,
                "current_batch128": current_batch128,
            },
            "accessIssues": [],
        },
        "sources": [compiled_source, implementation_source],
    }


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    (REPORT_DIR / "artifact.json").write_text(
        json.dumps(artifact(rows), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
