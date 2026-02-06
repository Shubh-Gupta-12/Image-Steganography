import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, NavLink, Link, useNavigate } from "react-router-dom";
import logo from "./logo.png";
import "./App.css";
import HomePage from "./pages/HomePage";
import ResultsPage from "./pages/ResultsPage";
import AboutPage from "./pages/AboutPage";
import ContactPage from "./pages/ContactPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { supabase, STORAGE_BUCKET } from "./lib/supabase";

function AppLayout() {
  const navigate = useNavigate();
  const { user, signOut } = useAuth();
  
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

  const handleSignOut = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (err) {
      console.error('Sign out error:', err);
    }
  };

  // Save image to Supabase storage
  const saveImageToStorage = async (blob, type) => {
    if (!user) return null;
    
    const timestamp = Date.now();
    const fileName = `${user.id}/${type}_${timestamp}.png`;
    
    const { data, error } = await supabase.storage
      .from(STORAGE_BUCKET)
      .upload(fileName, blob, {
        contentType: 'image/png',
        upsert: false
      });
    
    if (error) {
      console.error('Storage upload error:', error);
      return null;
    }
    
    // Get public URL
    const { data: urlData } = supabase.storage
      .from(STORAGE_BUCKET)
      .getPublicUrl(fileName);
    
    return urlData?.publicUrl || null;
  };

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

      // Save stego image to storage if user is logged in
      if (user) {
        setStatus("Saving to your account...");
        await saveImageToStorage(stegoBlob, 'stego');
      }

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
            {user ? (
              <>
                <span className="user-email">{user.email?.split('@')[0]}</span>
                <button className="nav-btn nav-btn-ghost" type="button" onClick={handleSignOut}>Logout</button>
              </>
            ) : (
              <>
                <NavLink to="/login" className={({ isActive }) => isActive ? "nav-btn nav-btn-ghost active" : "nav-btn nav-btn-ghost"}>Login</NavLink>
                <NavLink to="/signup" className="nav-btn nav-btn-solid">Sign Up</NavLink>
              </>
            )}
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
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
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
      <AuthProvider>
        <AppLayout />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
