// Morphling docs — small vanilla JS for theme toggle, mobile menu,
// search modal, and TOC scrollspy.

(function () {
  const root = document.documentElement;

  // ---- Theme toggle (persists user override; defaults to OS preference)
  const STORAGE_KEY = "morphling-docs-theme";
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") {
    root.setAttribute("data-theme", stored);
  }

  function currentTheme() {
    const explicit = root.getAttribute("data-theme");
    if (explicit) return explicit;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem(STORAGE_KEY, next);
  }

  document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
    b.addEventListener("click", toggleTheme);
  });

  // ---- Mobile sidebar
  document.querySelectorAll("[data-menu-toggle]").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelector(".sidebar")?.classList.toggle("is-open");
    });
  });

  // ---- Search modal
  const overlay = document.getElementById("search-overlay");
  const input = document.getElementById("search-input");
  const results = document.getElementById("search-results");

  // Static index — title, crumb, href
  const INDEX = [
    { t: "Welcome to Morphling", c: "Get Started", h: "index.html" },
    { t: "Installation", c: "Get Started", h: "install.html" },
    { t: "Quickstart", c: "Get Started", h: "quickstart.html" },
    { t: "Per-GEMM Green Contexts", c: "Internals", h: "green-context.html" },
    { t: "Worker Pool & Scheduling", c: "Internals", h: "worker-pool.html" },
    { t: "Virtual & Physical Deployment", c: "Deployment", h: "deployment.html" },
    { t: "Troubleshooting", c: "Reference", h: "troubleshooting.html" },
    { t: "EdgeSys '26 Paper", c: "Reference", h: "paper.html" },
  ];

  function openSearch() {
    if (!overlay) return;
    overlay.classList.add("is-open");
    input.value = "";
    renderResults("");
    setTimeout(() => input.focus(), 10);
  }
  function closeSearch() {
    overlay?.classList.remove("is-open");
  }
  function renderResults(q) {
    if (!results) return;
    const ql = q.trim().toLowerCase();
    const items = ql
      ? INDEX.filter((i) =>
          (i.t + " " + i.c).toLowerCase().includes(ql)
        )
      : INDEX;
    results.innerHTML = items
      .map(
        (i) =>
          `<a class="row" href="${i.h}"><span class="crumb">${i.c}</span><strong>${i.t}</strong></a>`
      )
      .join("");
  }

  document.querySelectorAll("[data-search-open]").forEach((b) => {
    b.addEventListener("click", openSearch);
  });
  overlay?.addEventListener("click", (e) => {
    if (e.target === overlay) closeSearch();
  });
  input?.addEventListener("input", (e) => renderResults(e.target.value));

  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openSearch();
    } else if (e.key === "Escape") {
      closeSearch();
    }
  });

  // ---- TOC scrollspy
  const tocLinks = Array.from(
    document.querySelectorAll("aside.toc a[href^='#']")
  );
  if (tocLinks.length) {
    const headings = tocLinks
      .map((a) => document.getElementById(a.getAttribute("href").slice(1)))
      .filter(Boolean);

    const setActive = (id) => {
      tocLinks.forEach((a) => {
        a.classList.toggle(
          "is-active",
          a.getAttribute("href") === "#" + id
        );
      });
    };

    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      {
        rootMargin: "-20% 0% -65% 0%",
        threshold: [0, 1],
      }
    );
    headings.forEach((h) => {
      io.observe(h);
    });
  }
})();
