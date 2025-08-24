export default function getValidToken() {
  const itemStr = localStorage.getItem('token');
  if (!itemStr) return null;

  try {
    const item = JSON.parse(itemStr);
    const now = new Date();

    if (now.getTime() > item.expiry) {
      localStorage.removeItem('token');
      return null;
    }

    return item.token;
  } catch {
    localStorage.removeItem('token');
    return null;
  }
}