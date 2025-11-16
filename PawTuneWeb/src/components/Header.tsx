import { Link } from "react-router-dom";
import "./Header.css";

export default function Header() {
  return (
    <header className="header">
      <Link to="/" className="header-logo">
        🐾 멍플리
      </Link>
      <nav className="header-nav">
        <Link to="/upload-dog" className="nav-link">
          업로드
        </Link>
      </nav>
    </header>
  );
}
