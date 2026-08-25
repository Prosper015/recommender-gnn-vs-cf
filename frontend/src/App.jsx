import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { Film, User, Network, BarChart3, Users, RefreshCw, AlertCircle } from "lucide-react";
import "./index.css";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [userId, setUserId] = useState(1);
  const [users] = useState([1, 2, 3, 4, 5]);
  const [recommendations, setRecommendations] = useState({ lightgcn: [], svd: [], item_item: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${API_URL}/api/v1/recommendations/${userId}?top_n=5`);
      setRecommendations({
        lightgcn: Array.isArray(res.data?.lightgcn) ? res.data.lightgcn : [],
        svd: Array.isArray(res.data?.svd) ? res.data.svd : [],
        item_item: Array.isArray(res.data?.item_item) ? res.data.item_item : [],
      });
    } catch {
      setError("Impossible de contacter l'API FastAPI (http://127.0.0.1:8000).");
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div className="container">
      <header className="header">
        <div className="header-title">
          <Film style={{ color: "var(--blue-primary)" }} size={28} />
          <h1>Système de Recommandation de Films</h1>
        </div>

        <div className="user-selector">
          <User size={18} style={{ color: "var(--text-secondary)" }} />
          <label htmlFor="user-select">Utilisateur :</label>
          <select
            id="user-select"
            className="select-input"
            value={userId}
            onChange={(e) => setUserId(Number(e.target.value))}
          >
            {users.map((id) => (
              <option key={id} value={id}>Utilisateur {id}</option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={fetchData} disabled={loading}>
            <RefreshCw size={16} className={loading ? "spin" : ""} />
            Actualiser
          </button>
        </div>
      </header>

      {error && (
        <div className="alert-error">
          <AlertCircle size={18} style={{ display: "inline", marginRight: "8px" }} />
          {error}
        </div>
      )}

      <main className="models-grid">
        {/* Colonne LightGCN */}
        <section className="column-card">
          <div className="column-header">
            <Network size={22} style={{ color: "var(--blue-primary)" }} />
            <h2>LightGCN (GNN)</h2>
            <span className="badge badge-lightgcn" style={{ marginLeft: "auto" }}>Graph</span>
          </div>
          <div className="movies-list">
            {recommendations.lightgcn && recommendations.lightgcn.length > 0 ? (
              recommendations.lightgcn.map((item) => (
                <div key={item.movieId} className="movie-card">
                  <div className="movie-title">{item.title}</div>
                  <div className="movie-meta">
                    <span>{item.genres}</span>
                    <strong>{item.score}</strong>
                  </div>
                </div>
              ))
            ) : (
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", textAlign: "center", padding: "12px" }}>
                Aucune recommandation disponible.
              </p>
            )}
          </div>
        </section>

        {/* Colonne SVD */}
        <section className="column-card">
          <div className="column-header">
            <BarChart3 size={22} style={{ color: "var(--green-primary)" }} />
            <h2>SVD (Matrice)</h2>
            <span className="badge badge-svd" style={{ marginLeft: "auto" }}>Factorisation</span>
          </div>
          <div className="movies-list">
            {recommendations.svd && recommendations.svd.length > 0 ? (
              recommendations.svd.map((item) => (
                <div key={item.movieId} className="movie-card">
                  <div className="movie-title">{item.title}</div>
                  <div className="movie-meta">
                    <span>{item.genres}</span>
                    <strong>{item.score}</strong>
                  </div>
                </div>
              ))
            ) : (
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", textAlign: "center", padding: "12px" }}>
                Aucune recommandation disponible.
              </p>
            )}
          </div>
        </section>

        {/* Colonne Item-Item CF */}
        <section className="column-card">
          <div className="column-header">
            <Users size={22} style={{ color: "var(--amber-primary)" }} />
            <h2>Item-Item CF</h2>
            <span className="badge badge-item" style={{ marginLeft: "auto" }}>Similarité</span>
          </div>
          <div className="movies-list">
            {recommendations.item_item && recommendations.item_item.length > 0 ? (
              recommendations.item_item.map((item) => (
                <div key={item.movieId} className="movie-card">
                  <div className="movie-title">{item.title}</div>
                  <div className="movie-meta">
                    <span>{item.genres}</span>
                    <strong>{item.score}</strong>
                  </div>
                </div>
              ))
            ) : (
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", textAlign: "center", padding: "12px" }}>
                Aucune recommandation disponible.
              </p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}