import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, NavLink, Link, useNavigate } from "react-router-dom";
import logo from "./logo.png";
import "./App.css";
import HomePage from "./pages/HomePage";
import ResultsPage from "./pages/ResultsPage";
import AboutPage from "./pages/AboutPage";
import ContactPage from "./pages/ContactPage";

function AppLayout() {
  const navigate = useNavigate();
  // Theme fixed to light mode
  
  // State for Processing
  const [coverImage, setCoverImage] = useState(null);
  const [secretImage, setSecretImage] = useState(null);
  const [alpha, setAlpha] = useState(0.1);
  const [stegoImage, setStegoImage] = useState(null);
  const [decodedImage, setDecodedImage] = useState(null);
  const [status, setStatus] = useState("Upload a cover image and a secret image to proceed.");
  const [loading, setLoading] = useState(false);

  // In production, API is on same origin at /api. In dev, use localhost:8000
  const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

  // Initialize Theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", "light");
    localStorage.removeItem("theme");
  }, []);

  const handleImageChange = (type) => (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (type === "cover") {
      setCoverImage(file);
    } else {
      setSecretImage(file);
    }
    
    // Determine the status based on update
    const hasCover = type === "cover" ? file : coverImage;
    const hasSecret = type === "secret" ? file : secretImage;

    if (hasCover && hasSecret) {
      setStatus("Ready to process.");
    } else if (hasCover && !hasSecret) {
      setStatus("Upload a secret image to proceed.");
    } else if (!hasCover && hasSecret) {
      setStatus("Upload a cover image to proceed.");
    }
  };

  const handleProcess = async () => {
    if (!coverImage || !secretImage) return;

    setLoading(true);
    setStatus("Encoding...");

    try {
      const formData = new FormData();
      formData.append("cover", coverImage);
      formData.append("secret", secretImage);
      formData.append("alpha", alpha);

      const encodeResponse = await fetch(`${API_BASE}/api/encode`, {
        method: "POST",
        body: formData,
      });

      if (!encodeResponse.ok) {
        const errorText = await encodeResponse.text();
        throw new Error(`Encode failed: ${errorText || encodeResponse.status}`);
      }

      const stegoBlob = await encodeResponse.blob();
      setStegoImage(URL.createObjectURL(stegoBlob));

      setStatus("Decoding...");

      const decodeForm = new FormData();
      decodeForm.append("cover", coverImage);
      decodeForm.append("stego", new File([stegoBlob], "stego.png", { type: "image/png" }));
      decodeForm.append("alpha", alpha);

      const decodeResponse = await fetch(`${API_BASE}/api/decode`, {
        method: "POST",
        body: decodeForm,
      });

      if (!decodeResponse.ok) {
        throw new Error("Decode failed.");
      }

      const decodedBlob = await decodeResponse.blob();
      setDecodedImage(URL.createObjectURL(decodedBlob));

      setStatus("Done!");
      navigate("/results");
    } catch (error) {
      setStatus(error.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <Link to="/" className="brand">
          <img src={logo} alt="Stegora logo" className="logo" />
        </Link>
        <div className="nav-right">
          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")} end>
              Home
            </NavLink>
            <NavLink to="/results" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
              Results
            </NavLink>
            <NavLink to="/about" className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}>
              About
            </NavLink>
          </nav>
          <div className="nav-actions">
            <NavLink to="/contact" className={({ isActive }) => isActive ? "nav-btn nav-btn-ghost active" : "nav-btn nav-btn-ghost"}>Contact</NavLink>
            <button className="nav-btn nav-btn-ghost" type="button">Login</button>
            <button className="nav-btn nav-btn-solid" type="button">Sign Up</button>
          </div>
          
        </div>
      </header>

      <Routes>
        <Route
          path="/"
          element={
            <HomePage
              coverImage={coverImage}
              secretImage={secretImage}
              alpha={alpha}
              setAlpha={setAlpha}
              status={status}
              loading={loading}
              handleImageChange={handleImageChange}
              handleProcess={handleProcess}
            />
          }
        />
        <Route
          path="/results"
          element={
            <ResultsPage
              coverImage={coverImage}
              secretImage={secretImage}
              stegoImage={stegoImage}
              decodedImage={decodedImage}
            />
          }
        />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/contact" element={<ContactPage />} />
      </Routes>

      <footer className="footer">
        <p>© 2026 Stegora. All processing occurs locally in your browser.</p>
      </footer>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}

export default App;
