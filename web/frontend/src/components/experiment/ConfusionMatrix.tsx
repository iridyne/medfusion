import React, { useState, useEffect } from "react";
import { Card, Spin, Empty, Tooltip } from "antd";

interface ConfusionMatrixProps {
  experimentId: string;
}

interface MatrixData {
  classes: string[];
  matrix: number[][];
  total: number;
}

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ experimentId }) => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<MatrixData | null>(null);

  useEffect(() => {
    fetchConfusionMatrix();
  }, [experimentId]);

  const fetchConfusionMatrix = async () => {
    setLoading(true);
    try {
      // Mock data - replace with actual API call
      const mockData: MatrixData = {
        classes: ["Class 0", "Class 1", "Class 2", "Class 3"],
        matrix: [
          [145, 8, 3, 2],
          [5, 132, 7, 4],
          [2, 6, 128, 5],
          [3, 4, 8, 138],
        ],
        total: 600,
      };

      await new Promise((resolve) => setTimeout(resolve, 500));
      setData(mockData);
    } catch (error) {
      console.error("Failed to fetch confusion matrix:", error);
    } finally {
      setLoading(false);
    }
  };

  const getColorForValue = (value: number, rowTotal: number, isCorrect: boolean) => {
    const percentage = (value / rowTotal) * 100;

    if (isCorrect) {
      // Green gradient for correct predictions
      if (percentage >= 90) return "#237804";
      if (percentage >= 80) return "#389e0d";
      if (percentage >= 70) return "#52c41a";
      if (percentage >= 60) return "#73d13d";
      return "#95de64";
    } else {
      // Red/Orange gradient for misclassifications
      if (percentage >= 10) return "#cf1322";
      if (percentage >= 5) return "#f5222d";
      if (percentage >= 2) return "#ff4d4f";
      if (percentage >= 1) return "#ff7875";
      return "#ffa39e";
    }
  };

  const getTextColor = (value: number, rowTotal: number, isCorrect: boolean) => {
    const percentage = (value / rowTotal) * 100;
    return isCorrect && percentage >= 70 ? "#fff" : "#000";
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: "40px" }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!data) {
    return <Empty description="No confusion matrix data available" />;
  }

  const { classes, matrix } = data;

  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: "12px",
        }}
      >
        <thead>
          <tr>
            <th style={{ padding: "8px", border: "1px solid #d9d9d9", background: "#fafafa" }}>
              Predicted →<br />Actual ↓
            </th>
            {classes.map((cls) => (
              <th
                key={cls}
                style={{
                  padding: "8px",
                  border: "1px solid #d9d9d9",
                  background: "#fafafa",
                  fontWeight: 600,
                }}
              >
                {cls}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const rowTotal = row.reduce((sum, val) => sum + val, 0);
            return (
              <tr key={i}>
                <td
                  style={{
                    padding: "8px",
                    border: "1px solid #d9d9d9",
                    background: "#fafafa",
                    fontWeight: 600,
                  }}
                >
                  {classes[i]}
                </td>
                {row.map((value, j) => {
                  const isCorrect = i === j;
                  const percentage = ((value / rowTotal) * 100).toFixed(1);
                  const bgColor = getColorForValue(value, rowTotal, isCorrect);
                  const textColor = getTextColor(value, rowTotal, isCorrect);

                  return (
                    <Tooltip
                      key={j}
                      title={`${classes[i]} → ${classes[j]}: ${value} (${percentage}%)`}
                    >
                      <td
                        style={{
                          padding: "12px",
                          border: "1px solid #d9d9d9",
                          textAlign: "center",
                          background: bgColor,
                          color: textColor,
                          fontWeight: isCorrect ? 600 : 400,
                          cursor: "pointer",
                          transition: "all 0.3s",
                        }}
                      >
                        <div>{value}</div>
                        <div style={{ fontSize: "10px", opacity: 0.8 }}>
                          {percentage}%
                        </div>
                      </td>
                    </Tooltip>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{ marginTop: 16, fontSize: 12, color: "#8c8c8c" }}>
        <div>
          <span style={{ fontWeight: 500 }}>Legend:</span> Diagonal (green) = Correct
          predictions, Off-diagonal (red) = Misclassifications
        </div>
        <div>Total samples: {data.total}</div>
      </div>
    </div>
  );
};

export default ConfusionMatrix;
