import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { supabase, STORAGE_BUCKET } from '../lib/supabase';
import { Link } from 'react-router-dom';

export default function HistoryPage() {
  const { user } = useAuth();
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all'); // 'all', 'stego', 'decoded'

  useEffect(() => {
    if (user) {
      fetchImages();
    } else {
      setLoading(false);
    }
  }, [user]);

  const fetchImages = async () => {
    try {
      const { data, error } = await supabase.storage
        .from(STORAGE_BUCKET)
        .list(user.id, {
          limit: 100,
          sortBy: { column: 'created_at', order: 'desc' }
        });

      if (error) throw error;

      // Get public URLs for each image
      const imagesWithUrls = data.map(file => {
        const { data: urlData } = supabase.storage
          .from(STORAGE_BUCKET)
          .getPublicUrl(`${user.id}/${file.name}`);
        
        return {
          ...file,
          url: urlData.publicUrl,
          type: file.name.startsWith('stego_') ? 'stego' : 'decoded'
        };
      });

      setImages(imagesWithUrls);
    } catch (err) {
      console.error('Error fetching images:', err);
    } finally {
      setLoading(false);
    }
  };

  const deleteImage = async (fileName) => {
    try {
      const { error } = await supabase.storage
        .from(STORAGE_BUCKET)
        .remove([`${user.id}/${fileName}`]);

      if (error) throw error;

      // Remove from local state
      setImages(images.filter(img => img.name !== fileName));
    } catch (err) {
      console.error('Error deleting image:', err);
    }
  };

  const filteredImages = filter === 'all' 
    ? images 
    : images.filter(img => img.type === filter);

  if (!user) {
    return (
      <main className="container">
        <section className="history-empty">
          <h2>My Images</h2>
          <p>Please <Link to="/login">login</Link> to view your saved images.</p>
        </section>
      </main>
    );
  }

  return (
    <main className="container">
      <section className="history-header">
        <h2>My Images</h2>
        <div className="filter-buttons">
          <button 
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All ({images.length})
          </button>
          <button 
            className={`filter-btn ${filter === 'stego' ? 'active' : ''}`}
            onClick={() => setFilter('stego')}
          >
            Stego ({images.filter(i => i.type === 'stego').length})
          </button>
          <button 
            className={`filter-btn ${filter === 'decoded' ? 'active' : ''}`}
            onClick={() => setFilter('decoded')}
          >
            Decoded ({images.filter(i => i.type === 'decoded').length})
          </button>
        </div>
      </section>

      {loading ? (
        <div className="history-loading">Loading your images...</div>
      ) : filteredImages.length === 0 ? (
        <div className="history-empty">
          <p>No images found. Start encoding to save images here!</p>
          <Link to="/" className="nav-btn nav-btn-solid">Go to Home</Link>
        </div>
      ) : (
        <div className="history-grid">
          {filteredImages.map((image) => (
            <div key={image.name} className="history-card">
              <div className="history-image-wrapper">
                <img src={image.url} alt={image.name} className="history-image" />
                <span className={`history-badge ${image.type}`}>
                  {image.type === 'stego' ? 'Stego' : 'Decoded'}
                </span>
              </div>
              <div className="history-card-footer">
                <span className="history-date">
                  {new Date(image.created_at).toLocaleDateString()}
                </span>
                <div className="history-actions">
                  <a 
                    href={image.url} 
                    download={image.name}
                    className="history-btn download"
                    title="Download"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                  </a>
                  <button 
                    onClick={() => deleteImage(image.name)}
                    className="history-btn delete"
                    title="Delete"
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
