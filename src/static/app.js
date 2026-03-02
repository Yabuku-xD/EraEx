import { setGreeting } from './js/hero-greeting.js';
import './js/app.core.js';

try {
  setGreeting();
} catch (error) {
  // Non-fatal UI enhancement.
}
