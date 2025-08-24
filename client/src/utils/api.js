export async function fetchModels(serverUrl) {
  const res = await fetch(`${serverUrl}/api/models/`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return await res.json();
} 