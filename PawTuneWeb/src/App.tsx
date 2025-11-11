import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import Upload from "./pages/Upload";
import RecommendedSong from "./pages/RecommendedSong";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="upload-dog" element={<Upload />} />
        <Route path="RecommendedSong" element={<RecommendedSong />} />
      </Route>
    </Routes>
  );
}
