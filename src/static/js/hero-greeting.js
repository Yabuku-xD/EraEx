export function setGreeting() {
  const hour = new Date().getHours();
  let greeting = 'Good evening';
  if (hour >= 5 && hour < 12) {
    greeting = 'Good morning';
  } else if (hour >= 12 && hour < 18) {
    greeting = 'Good afternoon';
  }
  const el = document.getElementById('hero-greeting');
  if (el) el.textContent = `${greeting}, Yabuku.`;
}
