const fs = require('fs');

function ensureDir(path) {
  if (!fs.existsSync(path)) {
    fs.mkdirSync(path, { recursive: true });
  }
}

module.exports = {
  ensureDir
};
