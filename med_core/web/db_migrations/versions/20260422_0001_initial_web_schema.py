"""Initial MedFusion Web metadata schema.

Revision ID: 20260422_0001
Revises:
Create Date: 2026-04-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260422_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_info",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("data_path", sa.String(length=500), nullable=False),
        sa.Column("dataset_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("num_samples", sa.Integer(), nullable=True),
        sa.Column("num_classes", sa.Integer(), nullable=True),
        sa.Column("train_samples", sa.Integer(), nullable=True),
        sa.Column("val_samples", sa.Integer(), nullable=True),
        sa.Column("test_samples", sa.Integer(), nullable=True),
        sa.Column("class_distribution", sa.JSON(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("analysis", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_dataset_info_created_at", "dataset_info", ["created_at"])
    op.create_index("ix_dataset_info_dataset_type", "dataset_info", ["dataset_type"])
    op.create_index("ix_dataset_info_name", "dataset_info", ["name"])
    op.create_index("ix_dataset_info_status", "dataset_info", ["status"])

    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("best_metric", sa.String(length=100), nullable=True),
        sa.Column("best_value", sa.String(length=100), nullable=True),
        sa.Column("output_dir", sa.String(length=500), nullable=True),
        sa.Column("checkpoint_path", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiments_created_at", "experiments", ["created_at"])
    op.create_index("ix_experiments_name", "experiments", ["name"])
    op.create_index("ix_experiments_status", "experiments", ["status"])

    op.create_table(
        "model_info",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("architecture", sa.String(length=100), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("num_parameters", sa.Integer(), nullable=True),
        sa.Column("model_size_mb", sa.Float(), nullable=True),
        sa.Column("checkpoint_path", sa.String(length=500), nullable=False),
        sa.Column("config_path", sa.String(length=500), nullable=True),
        sa.Column("trained_epochs", sa.Integer(), nullable=True),
        sa.Column("training_time", sa.Float(), nullable=True),
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
        sa.Column("num_classes", sa.Integer(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_info_created_at", "model_info", ["created_at"])
    op.create_index("ix_model_info_name", "model_info", ["name"])

    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(length=100), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("current_epoch", sa.Integer(), nullable=True),
        sa.Column("total_epochs", sa.Integer(), nullable=False),
        sa.Column("current_batch", sa.Integer(), nullable=True),
        sa.Column("total_batches", sa.Integer(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("current_loss", sa.Float(), nullable=True),
        sa.Column("current_accuracy", sa.Float(), nullable=True),
        sa.Column("current_lr", sa.Float(), nullable=True),
        sa.Column("best_loss", sa.Float(), nullable=True),
        sa.Column("best_accuracy", sa.Float(), nullable=True),
        sa.Column("best_epoch", sa.Integer(), nullable=True),
        sa.Column("gpu_usage", sa.Float(), nullable=True),
        sa.Column("gpu_memory", sa.Float(), nullable=True),
        sa.Column("cpu_usage", sa.Float(), nullable=True),
        sa.Column("ram_usage", sa.Float(), nullable=True),
        sa.Column("output_dir", sa.String(length=500), nullable=True),
        sa.Column("log_file", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_training_jobs_created_at", "training_jobs", ["created_at"])
    op.create_index("ix_training_jobs_job_id", "training_jobs", ["job_id"], unique=True)
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_training_jobs_status", table_name="training_jobs")
    op.drop_index("ix_training_jobs_job_id", table_name="training_jobs")
    op.drop_index("ix_training_jobs_created_at", table_name="training_jobs")
    op.drop_table("training_jobs")

    op.drop_index("ix_model_info_name", table_name="model_info")
    op.drop_index("ix_model_info_created_at", table_name="model_info")
    op.drop_table("model_info")

    op.drop_index("ix_experiments_status", table_name="experiments")
    op.drop_index("ix_experiments_name", table_name="experiments")
    op.drop_index("ix_experiments_created_at", table_name="experiments")
    op.drop_table("experiments")

    op.drop_index("ix_dataset_info_status", table_name="dataset_info")
    op.drop_index("ix_dataset_info_name", table_name="dataset_info")
    op.drop_index("ix_dataset_info_dataset_type", table_name="dataset_info")
    op.drop_index("ix_dataset_info_created_at", table_name="dataset_info")
    op.drop_table("dataset_info")

