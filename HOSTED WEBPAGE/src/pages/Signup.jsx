import React, { useState } from 'react';
import { createUserWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../firebase';
import Header from '../components/Header';
import Footer from '../components/Footer';

const Signup = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  const handleSignup = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    try {
      await createUserWithEmailAndPassword(auth, email, password);
      window.location.href = '/'; // redirect to homepage
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <Header />
      <main className="flex flex-col items-center justify-center flex-grow p-4">
        <h2 className="text-3xl font-bold mb-4">Join the Movement</h2>
        <p className="text-gray-400 mb-4">Break communication barriers with AI</p>

        {error && <p className="text-red-500 mb-2">{error}</p>}

        <form onSubmit={handleSignup} className="w-full max-w-sm">
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" className="w-full p-2 mb-3 rounded bg-gray-800 text-white border border-gray-600" required />
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" className="w-full p-2 mb-3 rounded bg-gray-800 text-white border border-gray-600" required />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" className="w-full p-2 mb-3 rounded bg-gray-800 text-white border border-gray-600" required />
          <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Confirm Password" className="w-full p-2 mb-4 rounded bg-gray-800 text-white border border-gray-600" required />
          <button type="submit" className="w-full bg-red-600 hover:bg-red-700 text-white p-2 rounded">Create Account</button>
        </form>

        <p className="mt-4 text-sm text-gray-400">
          Already have an account? <a href="/login" className="text-red-500 hover:underline">Log in</a>
        </p>
      </main>
      <Footer />
    </div>
  );
};

export default Signup;
