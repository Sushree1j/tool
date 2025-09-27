require('dotenv').config();
const path = require('path');
const app = require('./app');
const { ensureDir } = require('./utils/ensureDir');
const { getDb } = require('./db');

const PORT = Number(process.env.PORT) || 4000;
const uploadsDir = path.join(__dirname, '..', 'uploads');

ensureDir(uploadsDir);
getDb();

app.listen(PORT, () => {
  console.log(`NovaShare server listening on port ${PORT}`);
});
