import React from "react";
import { ArrowRight } from "lucide-react";

const GlassCTA = () => {
  return (
    <div className="relative mx-auto mt-20 mb-5 px-4 w-full max-w-md backdrop-blur-md bg-white/10 border border-white/20 rounded-full shadow-lg flex items-center justify-between text-white py-2 px-4 overflow-hidden">
      <p className="text-xs sm:text-sm font-medium truncate z-10">
        We are the <span className="font-bold">voice</span> of the unfortunate
      </p>
      <button className="ml-2 bg-gradient-to-r from-red-500 to-red-700 hover:opacity-90 p-1.5 rounded-full transition z-10">
        <ArrowRight size={16} />
      </button>
    </div>
  );
};

export default GlassCTA;