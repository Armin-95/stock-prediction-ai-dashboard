document.addEventListener("DOMContentLoaded", () => {
  const banner = document.getElementById("cookie-banner");
  const acceptBtn = document.getElementById("accept-cookies");
  const rejectBtn = document.getElementById("reject-cookies");

  if (!banner || !acceptBtn || !rejectBtn) {
    return;
  }

  const hasConsentCookie = document.cookie
    .split("; ")
    .some(cookie => cookie.startsWith("cookie_consent="));

  if (!hasConsentCookie) {
    banner.style.display = "block";
  }

  acceptBtn.addEventListener("click", () => {
    document.cookie = "cookie_consent=accepted; path=/; max-age=31536000; SameSite=Lax";
    banner.style.display = "none";
  });

  rejectBtn.addEventListener("click", () => {
    document.cookie = "cookie_consent=rejected; path=/; max-age=31536000; SameSite=Lax";
    banner.style.display = "none";
  });
});