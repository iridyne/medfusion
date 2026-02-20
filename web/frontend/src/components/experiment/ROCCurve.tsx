import React, { useState, useEffect } from "react";
import { Spin, Empty, Select, Space } from "antd";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface ROCCurveProps {
  experimentIds: string[];
}

interface ROCData {
  experimentId: string;
  experimentName: string;
  auc: number;
  points: Array<{
    fpr: number;
    tpr: number;
    threshold: number;
  }>;
}

const ROCCurve: React.FC<ROCCurveProps> = ({ experimentIds }) => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ROCData[]>([]);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);

  useEffect(() => {
    if (experimentIds.length > 0) {
      fetchROCData();
      setSelectedExperiments(experimentIds.slice(0, 4)); // Show max 4 by default
    }
  }, [experimentIds]);

  const fetchROCData = async () => {
    setLoading(true);
    try {
      // Mock data - replace with actual API call
      const mockData: ROCData[] = experimentIds.map((id, index) => {
        // Generate realistic ROC curve points
        const points = [];
        const numPoints = 50;

        // Different AUC values for different experiments
        const baseAUC = 0.85 + index * 0.05;

        for (let i = 0; i <= numPoints; i++) {
          const fpr = i / numPoints;
          // Generate TPR with some randomness but maintaining AUC
          let tpr = fpr + (baseAUC - 0.5) * 2 * (1 - fpr);
          tpr = Math.min(1, Math.max(0, tpr + (Math.random() - 0.5) * 0.05));

          points.push({
            fpr: parseFloat(fpr.toFixed(3)),
            tpr: parseFloat(tpr.toFixed(3)),
            threshold: parseFloat((1 - i / numPoints).toFixed(3)),
          });
        }

        return {
          experimentId: id,
          experimentName: `Experiment ${index + 1}`,
          auc: parseFloat(baseAUC.toFixed(3)),
          points,
        };
      });

      await new Promise((resolve) => setTimeout(resolve, 500));
      setData(mockData);
    } catch (error) {
      console.error("Failed to fetch ROC data:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "40px" }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!data || data.length === 0) {
    return <Empty description="No ROC curve data available" />;
  }

  // Prepare chart data by merging all experiments
  const chartData: any[] = [];
  const maxPoints = Math.max(...data.map((d) => d.points.length));

  for (let i = 0; i < maxPoints; i++) {
    const point: any = {};

    selectedExperiments.forEach((expId) => {
      const expData = data.find((d) => d.experimentId === expId);
      if (expData && expData.points[i]) {
        point.fpr = expData.points[i].fpr;
        point[`${expData.experimentName}_tpr`] = expData.points[i].tpr;
      }
    });

    if (Object.keys(point).length > 1) {
      chartData.push(point);
    }
  }

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

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <span>Select experiments to display:</span>
        <Select
          mode="multiple"
          style={{ minWidth: 300 }}
          placeholder="Select experiments"
          value={selectedExperiments}
          onChange={setSelectedExperiments}
          maxTagCount={3}
        >
          {data.map((exp) => (
            <Select.Option key={exp.experimentId} value={exp.experimentId}>
              {exp.experimentName} (AUC: {exp.auc.toFixed(3)})
            </Select.Option>
          ))}
        </Select>
      </Space>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            label={{
              value: "False Positive Rate",
              position: "insideBottom",
              offset: -10,
            }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            label={{
              value: "True Positive Rate",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip
            formatter={(value: number, name: string) => [
              value.toFixed(3),
              name.replace("_tpr", ""),
            ]}
            labelFormatter={(label) => `FPR: ${parseFloat(label).toFixed(3)}`}
          />
          <Legend
            wrapperStyle={{ paddingTop: 20 }}
            formatter={(value) => {
              const expData = data.find((d) =>
                value.includes(d.experimentName)
              );
              return expData
                ? `${expData.experimentName} (AUC: ${expData.auc.toFixed(3)})`
                : value;
            }}
          />

          {/* Diagonal reference line (random classifier) */}
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke="#d9d9d9"
            strokeDasharray="5 5"
            label={{
              value: "Random (AUC=0.5)",
              position: "insideBottomRight",
              fill: "#8c8c8c",
              fontSize: 12,
            }}
          />

          {selectedExperiments.map((expId, index) => {
            const expData = data.find((d) => d.experimentId === expId);
            if (!expData) return null;

            return (
              <Line
                key={expId}
                type="monotone"
                dataKey={`${expData.experimentName}_tpr`}
                stroke={colors[index % colors.length]}
                strokeWidth={2}
                dot={false}
                name={expData.experimentName}
              />
            );
          })}
        </LineChart>
      </ResponsiveContainer>

      <div style={{ marginTop: 16, fontSize: 12, color: "#8c8c8c" }}>
        <div>
          <span style={{ fontWeight: 500 }}>ROC Curve:</span> Receiver Operating
          Characteristic curve showing the trade-off between True Positive Rate and
          False Positive Rate at various threshold settings.
        </div>
        <div style={{ marginTop: 8 }}>
          <span style={{ fontWeight: 500 }}>AUC:</span> Area Under the Curve - Higher
          is better (1.0 = perfect classifier, 0.5 = random classifier)
        </div>
      </div>
    </div>
  );
};

export default ROCCurve;
