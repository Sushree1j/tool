const path = require('path');
const Database = require('better-sqlite3');
const { ensureDir } = require('./utils/ensureDir');

const dataDir = path.join(__dirname, '..', 'data');
const dbPath = path.join(dataDir, 'fileshare.db');
const isTest = process.env.NODE_ENV === 'test';

let dbInstance;

function getDb() {
  if (!dbInstance) {
    if (isTest) {
      dbInstance = new Database(':memory:');
    } else {
      ensureDir(dataDir);
      dbInstance = new Database(dbPath);
    }
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

function closeDb() {
  if (dbInstance) {
    dbInstance.close();
    dbInstance = undefined;
  }
}

module.exports = {
  getDb,
  closeDb,
  dbPath
};
