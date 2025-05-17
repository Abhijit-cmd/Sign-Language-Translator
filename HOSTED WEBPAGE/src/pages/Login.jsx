import React, { useState } from 'react';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebase';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
const handleGoogleLogin = async () => {
  const provider = new GoogleAuthProvider();
  try {
    await signInWithPopup(auth, provider);
    window.location.href = "/";
  } catch (err) {
    setError("Google sign-in failed.");
  }
};


const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await signInWithEmailAndPassword(auth, email, password);
      window.location.href = "/"; // redirect to home
    } catch (err) {
      setError("Invalid email or password.");
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <Header />
      <main className="flex flex-col items-center justify-center flex-grow p-4">
        <h2 className="text-3xl font-bold mb-4">Welcome Back</h2>
        {error && <p className="text-red-500 mb-2">{error}</p>}
        <form onSubmit={handleLogin} className="w-full max-w-sm">
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" className="w-full p-2 mb-3 rounded bg-gray-800 text-white border border-gray-600" required />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" className="w-full p-2 mb-4 rounded bg-gray-800 text-white border border-gray-600" required />
          <button type="submit" className="w-full bg-red-600 hover:bg-red-700 text-white p-2 rounded">Login</button>
        </form>
        <p className="mt-4 text-sm text-gray-400">
          New here? <a href="/signup" className="text-red-500 hover:underline">Sign up instead</a>
        </p>
      </main>
      <button
  onClick={handleGoogleLogin}
  className="w-full mt-4 bg-white text-black hover:bg-gray-100 p-2 rounded shadow flex items-center justify-center gap-2"
>
  <img src="https://www.svgrepo.com/show/355037/google.svg" alt="Google" className="w-5 h-5" />
  Continue with Google
</button>

      <Footer />
    </div>
  );
};

export default Login;
