import earthImage from "figma:asset/359aa392ba2ec6a07d1fa277ba543ac4b8b6873c.png";

export function EarthScene() {
  return (
    <div className="relative mx-auto perspective-1000" style={{ width: 500, height: 500 }}>
      {/* Earth Container */}
      <div className="absolute inset-0 earth-container">
        {/* Earth Globe (static) */}
        <div className="relative w-full h-full rounded-full overflow-hidden shadow-2xl shadow-blue-500/30">
          <img
            src={earthImage}
            alt="Earth from space"
            className="w-full h-full object-cover scale-110"
          />
          {/* Remove glistening/shining overlays for a cleaner globe */}
        </div>
        
        {/* Subtle atmosphere only */}
        <div className="absolute inset-0 rounded-full bg-blue-400/10 blur-xl"></div>
      </div>

      {/* Static, realistic satellites placed within the globe (SVG) */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Satellite A */}
        <div className="absolute" style={{ left: '22%', top: '30%', transform: 'translate(-50%, -50%) rotate(-12deg)' }}>
          <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="28" y="26" width="8" height="12" rx="1.5" fill="#cbd5e1"/>
            <rect x="20" y="24" width="6" height="16" rx="1" fill="#94a3b8"/>
            <rect x="38" y="24" width="6" height="16" rx="1" fill="#94a3b8"/>
            <rect x="6" y="28" width="14" height="8" rx="1.2" fill="#22d3ee"/>
            <rect x="44" y="28" width="14" height="8" rx="1.2" fill="#22d3ee"/>
          </svg>
        </div>
        {/* Satellite B */}
        <div className="absolute" style={{ left: '76%', top: '44%', transform: 'translate(-50%, -50%) rotate(8deg)' }}>
          <svg width="54" height="54" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="28" y="26" width="8" height="12" rx="1.5" fill="#cbd5e1"/>
            <rect x="19" y="23" width="7" height="18" rx="1" fill="#a3b1c6"/>
            <rect x="38" y="23" width="7" height="18" rx="1" fill="#a3b1c6"/>
            <rect x="5" y="27" width="15" height="10" rx="1.4" fill="#34d399"/>
            <rect x="44" y="27" width="15" height="10" rx="1.4" fill="#34d399"/>
          </svg>
        </div>
        {/* Satellite C */}
        <div className="absolute" style={{ left: '38%', top: '78%', transform: 'translate(-50%, -50%) rotate(-18deg)' }}>
          <svg width="50" height="50" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="28" y="26" width="8" height="12" rx="1.5" fill="#cbd5e1"/>
            <rect x="20" y="24" width="6" height="16" rx="1" fill="#94a3b8"/>
            <rect x="38" y="24" width="6" height="16" rx="1" fill="#94a3b8"/>
            <rect x="6" y="28" width="14" height="8" rx="1.2" fill="#a78bfa"/>
            <rect x="44" y="28" width="14" height="8" rx="1.2" fill="#a78bfa"/>
          </svg>
        </div>
      </div>

      {/* Orbital Path Indicators */}
      <div className="absolute inset-4 pointer-events-none">
        <div className="w-full h-full border border-cyan-400/15 rounded-full orbit-path-glow"></div>
      </div>
      <div className="absolute -inset-4 pointer-events-none">
        <div className="w-full h-full border border-purple-400/15 rounded-full orbit-path-glow"></div>
      </div>

      {/* Starfield Background */}
      <div className="absolute -inset-20 pointer-events-none overflow-hidden">
        {Array.from({ length: 30 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-white/60 rounded-full star-twinkle"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

    </div>
  );
}