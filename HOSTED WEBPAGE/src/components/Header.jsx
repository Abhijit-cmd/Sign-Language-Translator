import React, { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "../firebase"; // make sure this path is correct

const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [userInitial, setUserInitial] = useState("");

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        const name = user.displayName || user.email;
        setUserInitial(name?.charAt(0).toUpperCase());
      } else {
        setUserInitial("");
      }
    });
    return () => unsubscribe();
  }, []);

  const handleLogout = async () => {
    await signOut(auth);
    setUserInitial("");
    window.location.href = "/"; // optional redirect
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 px-4 sm:px-6 md:px-12 py-4 mb-4 md:mb-0">
      <div className="flex justify-between items-center backdrop-blur-md bg-white/10 border border-white/20 rounded-full shadow-md px-6 py-3 max-w-7xl mx-auto">
        {/* Logo */}
        <div className="text-2xl font-bold text-white">
          <span className="text-red-500">S</span>ignLang
          <span className="text-sm text-white/70"> AI</span>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex space-x-6 text-white text-sm font-medium">
          <a href="/" className="hover:text-red-400 transition">Home</a>
          <a href="/about" className="hover:text-red-400 transition">About</a>
          <a href="/services" className="hover:text-red-400 transition">Services</a>
          <a href="/features" className="hover:text-red-400 transition">Features</a>
          <a href="/pricing" className="hover:text-red-400 transition">Pricing</a>
          <a href="/works" className="hover:text-red-400 transition">Works</a>
        </nav>

        {/* Sign Up or Initial (Desktop) */}
        {userInitial ? (
          <div className="hidden md:flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-red-600 text-white flex items-center justify-center font-bold">
              {userInitial}
            </div>
            <button
              onClick={handleLogout}
              className="text-sm bg-white/10 hover:bg-white/20 text-white px-4 py-1.5 rounded-full transition"
            >
              Logout
            </button>
          </div>
        ) : (
          <a
            href="/signup"
            className="hidden md:inline-block bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-full shadow-lg transition"
          >
            Sign Up
          </a>
        )}

        {/* Mobile Menu Toggle */}
        <button
          className="md:hidden text-white"
          onClick={() => setMenuOpen(!menuOpen)}
        >
          {menuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Dropdown Menu */}
      {menuOpen && (
        <div className="md:hidden mt-4 backdrop-blur-md bg-white/10 border border-white/20 rounded-xl mx-4 shadow-md text-white text-center py-4 space-y-3">
          <a href="/" className="block hover:text-red-400 transition">Home</a>
          <a href="/about" className="block hover:text-red-400 transition">About</a>
          <a href="/services" className="block hover:text-red-400 transition">Services</a>
          <a href="/features" className="block hover:text-red-400 transition">Features</a>
          <a href="/pricing" className="block hover:text-red-400 transition">Pricing</a>
          <a href="/works" className="block hover:text-red-400 transition">Works</a>
          
          {userInitial ? (
            <>
              <div className="flex justify-center items-center gap-2">
                <div className="w-10 h-10 rounded-full bg-red-600 flex items-center justify-center font-bold text-white">
                  {userInitial}
                </div>
                <button
                  onClick={handleLogout}
                  className="bg-white/10 hover:bg-white/20 px-4 py-2 rounded-full text-sm transition"
                >
                  Logout
                </button>
              </div>
            </>
          ) : (
            <a
              href="/signup"
              className="inline-block bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-full shadow-lg transition mt-2"
            >
              Sign Up
            </a>
          )}
        </div>
      )}
    </header>
  );
};

export default Header;
