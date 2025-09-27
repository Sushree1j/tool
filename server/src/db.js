const path = require('path');
const Database = require('better-sqlite3');
const { ensureDir } = require('./utils/ensureDir');

const dataDir = path.join(__dirname, '..', 'data');
const dbPath = path.join(dataDir, 'fileshare.db');

let dbInstance;

function getDb() {
  if (!dbInstance) {
    ensureDir(dataDir);
    dbInstance = new Database(dbPath);
    dbInstance.pragma('journal_mode = WAL');
    dbInstance.exec(`
      CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        stored_name TEXT NOT NULL,
        size INTEGER NOT NULL,
        mime_type TEXT,
        checksum TEXT,
        download_count INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
      );
    `);
  }

  return dbInstance;
}

module.exports = {
  getDb,
  dbPath
};
