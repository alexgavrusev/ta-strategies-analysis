import pandas as pd

from config import (
    raw_results_file,
    all_table_file,
    top_10_table_file,
)


def prepare_table(stats: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Strategy": stats.index,
            "$\\widehat{\\text{SR}}$": stats["Ann_SR"],
            "$p^s$": stats["SR_p_value"],
            "$\\widehat{\\text{PSR}}$": stats["PSR"],
            "$\\widehat{\\text{DSR}}$": stats["DSR"],
            "$p^M$": stats["HLZ_p_value"],
        }
    )

    return (
        df.style.format_index(escape="latex")
        .format(escape="latex", precision=3)
        .hide(axis="index")
    )


if __name__ == "__main__":
    statistics = pd.read_csv(raw_results_file, index_col=0, header=0)

    prepare_table(statistics).to_latex(
        all_table_file, hrules=True, environment="longtable"
    )

    top_10_df = statistics.sort_values("DSR", ascending=False).head(10)

    prepare_table(top_10_df).to_latex(top_10_table_file, hrules=True)
