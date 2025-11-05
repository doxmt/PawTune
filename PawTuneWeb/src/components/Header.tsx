import { Link } from "react-router-dom";
import "./Header.css";

export default function Header() {
  return (
    <header className="header">
      <Link to="/" className="header-logo">
        🐾 PawTune
      </Link>
      <nav className="header-nav">
        <Link to="/" className="nav-link">
          홈
        </Link>
        <Link to="/upload" className="nav-link">
          업로드
        </Link>
        <Link to="/result" className="nav-link">
          결과
        </Link>
      </nav>
    </header>
  );
}
