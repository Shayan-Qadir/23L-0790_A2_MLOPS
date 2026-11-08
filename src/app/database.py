import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionLog(db.Model):
    """
    Task 5 Database Trail:
    Every request leaves a permanent, queryable record that answers:
    what was decided, when, by which model, and how confidently.
    """
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    filename = db.Column(db.String(255), nullable=False)
    decision = db.Column(db.String(50), nullable=False)  # "AI-Generated" or "Real Artwork"
    confidence = db.Column(db.Float, nullable=False)    # e.g., 0.945 (94.5%)
    raw_probability = db.Column(db.Float, nullable=False)
    threshold_used = db.Column(db.Float, nullable=False, default=0.50)
    model_version = db.Column(db.String(100), nullable=False, default="MobileNetV2_Champion_v1")
    inference_time_ms = db.Column(db.Float, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "filename": self.filename,
            "decision": self.decision,
            "confidence": round(self.confidence * 100, 2),
            "raw_probability": round(self.raw_probability, 4),
            "threshold_used": self.threshold_used,
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 2)
        }

def init_db(app):
    """Initializes database with the Flask application context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

def log_prediction(filename, decision, confidence, raw_probability, threshold_used, model_version, inference_time_ms):
    """Persists a new inference search log entry into the database."""
    entry = PredictionLog(
        filename=filename,
        decision=decision,
        confidence=confidence,
        raw_probability=raw_probability,
        threshold_used=threshold_used,
        model_version=model_version,
        inference_time_ms=inference_time_ms
    )
    db.session.add(entry)
    db.session.commit()
    return entry

def get_recent_history(limit=50):
    """Retrieves past prediction logs in reverse chronological order."""
    return PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(limit).all()

