const { randomUUID } = require('crypto');
const path = require('path');
const fs = require('fs');
const { getDb } = require('../db');

function insertFile({ code, name, storedName, size, mimeType, checksum }) {
  const db = getDb();
  const id = randomUUID();
  const stmt = db.prepare(`
    INSERT INTO files (id, code, name, stored_name, size, mime_type, checksum)
    VALUES (@id, @code, @name, @stored_name, @size, @mime_type, @checksum)
  `);

  stmt.run({
    id,
    code,
    name,
    stored_name: storedName,
    size,
    mime_type: mimeType ?? null,
    checksum: checksum ?? null
  });

  return findById(id);
}

function findById(id) {
  const db = getDb();
  return db.prepare('SELECT * FROM files WHERE id = ?').get(id);
}

function findByCode(code) {
  const db = getDb();
  return db.prepare('SELECT * FROM files WHERE code = ?').get(code);
}

function listRecent(limit = 25) {
  const db = getDb();
  return db
    .prepare('SELECT * FROM files ORDER BY datetime(created_at) DESC LIMIT ?')
    .all(limit);
}

function incrementDownload(code) {
  const db = getDb();
  const stmt = db.prepare('UPDATE files SET download_count = download_count + 1 WHERE code = ?');
  stmt.run(code);
}

function deleteByCode(code, uploadsDir) {
  const db = getDb();
  const file = findByCode(code);
  if (!file) return false;
  const filePath = path.join(uploadsDir, file.stored_name);
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
  const stmt = db.prepare('DELETE FROM files WHERE code = ?');
  stmt.run(code);
  return true;
}

module.exports = {
  insertFile,
  findById,
  findByCode,
  listRecent,
  incrementDownload,
  deleteByCode
};
