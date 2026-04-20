import React from "react";
import { Card, Empty } from "antd";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type {
  NameType,
  Payload,
  ValueType,
} from "recharts/types/component/DefaultTooltipContent";

interface ComparisonMetrics {
  metric: string;
  experiments: { [key: string]: number };
  best_experiment: string;
}

interface MetricsChartProps {
  data: ComparisonMetrics[];
  chartType: "bar" | "line";
  height?: number;
}

function getTooltipScalar(
  value: ValueType | undefined,
): number | string | undefined {
  const scalarValue = Array.isArray(value) ? value[0] : value;
  return typeof scalarValue === "number" || typeof scalarValue === "string"
    ? scalarValue
    : undefined;
}

function formatTooltipSeriesName(name: NameType | undefined): string {
  return name === undefined ? "" : String(name);
}

function formatMetricsTooltip(
  value: ValueType | undefined,
  name: NameType | undefined,
  item: Payload<ValueType, NameType>,
): [string, string] {
  const scalarValue = getTooltipScalar(value);
  const numericValue =
    typeof scalarValue === "number" ? scalarValue : Number(scalarValue ?? 0);
  const metricName = String((item.payload as { metric?: string }).metric ?? "");
  const suffix = metricName === "Loss" ? "" : "%";

  return [`${numericValue.toFixed(2)}${suffix}`, formatTooltipSeriesName(name)];
}

const MetricsChart: React.FC<MetricsChartProps> = ({
  data,
  chartType,
  height = 400,
}) => {
  if (!data || data.length === 0) {
    return <Empty description="No data to display" />;
  }

  // Transform data for recharts
  const chartData = data.map((item) => {
    const row: any = { metric: item.metric };
    Object.entries(item.experiments).forEach(([expName, value]) => {
      row[expName] = item.metric === "Loss" ? value : value * 100;
    });
    return row;
  });

  // Get experiment names (keys)
  const experimentNames = Object.keys(data[0].experiments);

  // Color palette for different experiments
  const colors = [
    "#1890ff",
    "#52c41a",
    "#faad14",
    "#f5222d",
    "#722ed1",
    "#13c2c2",
    "#eb2f96",
    "#fa8c16",
  ];

  if (chartType === "bar") {
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="metric" />
          <YAxis
            label={{
              value: "Value (%)",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip
            formatter={formatMetricsTooltip}
          />
          <Legend />
          {experimentNames.map((expName, index) => (
            <Bar
              key={expName}
              dataKey={expName}
              fill={colors[index % colors.length]}
              name={expName}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="metric" />
        <YAxis
          label={{
            value: "Value (%)",
            angle: -90,
            position: "insideLeft",
          }}
        />
        <Tooltip
          formatter={formatMetricsTooltip}
        />
        <Legend />
        {experimentNames.map((expName, index) => (
          <Line
            key={expName}
            type="monotone"
            dataKey={expName}
            stroke={colors[index % colors.length]}
            strokeWidth={2}
            name={expName}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default MetricsChart;
