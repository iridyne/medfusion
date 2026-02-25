#!/usr/bin/env python
"""
比较基准测试结果，检测性能回归

用于 CI/CD 流程中自动检测性能下降。
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(filepath):
    """加载基准测试结果"""
    with open(filepath) as f:
        return json.load(f)


def compare_results(baseline, current, tolerance=0.05):
    """
    比较两个基准测试结果

    Args:
        baseline: 基线结果
        current: 当前结果
        tolerance: 容忍度（默认 5%）

    Returns:
        (has_regression, comparisons)
    """
    has_regression = False
    comparisons = []

    def compare_dict(base_dict, curr_dict, path=""):
        """递归比较字典"""
        nonlocal has_regression

        for key in base_dict:
            if key not in curr_dict:
                continue

            base_val = base_dict[key]
            curr_val = curr_dict[key]

            if isinstance(base_val, dict) and isinstance(curr_val, dict):
                # 递归比较
                compare_dict(base_val, curr_val, f"{path}.{key}" if path else key)
            elif isinstance(base_val, (int, float)) and isinstance(
                curr_val, (int, float)
            ):
                # 比较数值
                if key == "throughput":
                    # 吞吐量：越高越好
                    change = (curr_val - base_val) / base_val
                    is_regression = change < -tolerance

                    comparison = {
                        "name": f"{path}.{key}" if path else key,
                        "baseline": base_val,
                        "current": curr_val,
                        "change": change * 100,
                        "is_regression": is_regression,
                    }

                    comparisons.append(comparison)

                    if is_regression:
                        has_regression = True

                elif key == "duration":
                    # 执行时间：越低越好
                    change = (curr_val - base_val) / base_val
                    is_regression = change > tolerance

                    comparison = {
                        "name": f"{path}.{key}" if path else key,
                        "baseline": base_val,
                        "current": curr_val,
                        "change": change * 100,
                        "is_regression": is_regression,
                    }

                    comparisons.append(comparison)

                    if is_regression:
                        has_regression = True

    compare_dict(baseline, current)

    return has_regression, comparisons


def print_comparison(comparisons, verbose=False):
    """打印比较结果"""
    print("\n" + "=" * 80)
    print("性能比较结果")
    print("=" * 80)

    # 按是否回归分组
    regressions = [c for c in comparisons if c["is_regression"]]
    improvements = [
        c for c in comparisons if c["change"] > 5 and not c["is_regression"]
    ]
    stable = [c for c in comparisons if abs(c["change"]) <= 5]

    # 打印回归
    if regressions:
        print("\n❌ 性能回归 (下降 > 5%):")
        print("-" * 80)
        for comp in regressions:
            print(f"  {comp['name']}:")
            print(f"    基线: {comp['baseline']:.2f}")
            print(f"    当前: {comp['current']:.2f}")
            print(f"    变化: {comp['change']:+.1f}%")
            print()

    # 打印改进
    if improvements:
        print("\n✅ 性能改进 (提升 > 5%):")
        print("-" * 80)
        for comp in improvements:
            print(f"  {comp['name']}:")
            print(f"    基线: {comp['baseline']:.2f}")
            print(f"    当前: {comp['current']:.2f}")
            print(f"    变化: {comp['change']:+.1f}%")
            print()

    # 打印稳定的（仅在 verbose 模式）
    if verbose and stable:
        print("\n➡️  性能稳定 (变化 ≤ 5%):")
        print("-" * 80)
        for comp in stable:
            print(f"  {comp['name']}: {comp['change']:+.1f}%")

    # 总结
    print("\n" + "=" * 80)
    print("总结:")
    print(f"  回归: {len(regressions)}")
    print(f"  改进: {len(improvements)}")
    print(f"  稳定: {len(stable)}")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较基准测试结果")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="基线结果文件",
    )
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="当前结果文件",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="容忍度（默认 0.05 = 5%%）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="如果有回归则返回非零退出码",
    )

    args = parser.parse_args()

    # 检查文件是否存在
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists():
        print(f"❌ 错误: 基线文件不存在: {baseline_path}")
        sys.exit(1)

    if not current_path.exists():
        print(f"❌ 错误: 当前结果文件不存在: {current_path}")
        sys.exit(1)

    # 加载结果
    print(f"\n加载基线: {baseline_path}")
    baseline = load_results(baseline_path)

    print(f"加载当前: {current_path}")
    current = load_results(current_path)

    # 比较
    has_regression, comparisons = compare_results(
        baseline,
        current,
        tolerance=args.tolerance,
    )

    # 打印结果
    print_comparison(comparisons, verbose=args.verbose)

    # 退出
    if has_regression:
        print("\n⚠️  检测到性能回归！")
        if args.fail_on_regression:
            sys.exit(1)
    else:
        print("\n✅ 没有性能回归")

    sys.exit(0)


if __name__ == "__main__":
    main()
