const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { ensureDir } = require('../utils/ensureDir');
const { generateShareCode } = require('../utils/generateCode');
const fileModel = require('../models/fileModel');

const TEN_GB = 10 * 1024 * 1024 * 1024;
const uploadsDir = path.join(__dirname, '..', '..', 'uploads');
ensureDir(uploadsDir);

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (_req, file, cb) => {
    const hash = crypto.createHash('sha1').update(`${Date.now()}-${file.originalname}`).digest('hex');
    const ext = path.extname(file.originalname);
    cb(null, `${hash}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: TEN_GB
  }
});

function buildShareLink(req, code) {
  const protocol = req.headers['x-forwarded-proto'] || req.protocol;
  const host = req.headers['x-forwarded-host'] || req.get('host');
  return `${protocol}://${host}${req.baseUrl}/${code}`;
}

function formatFile(file, req) {
  if (!file) return null;
  return {
    id: file.id,
    code: file.code,
    name: file.name,
    size: file.size,
    mimeType: file.mime_type,
    checksum: file.checksum,
    downloadCount: file.download_count,
    createdAt: file.created_at,
    link: req ? `${buildShareLink(req, file.code)}` : undefined
  };
}

const router = express.Router();

router.get('/', (req, res) => {
  const files = fileModel.listRecent(50).map((file) => formatFile(file));
  res.json({ files });
});

router.get('/:code', (req, res) => {
  const { code } = req.params;
  const file = fileModel.findByCode(code);
  if (!file) {
    return res.status(404).json({ message: 'File not found' });
  }
  return res.json({ file: formatFile(file, req) });
});

router.get('/:code/download', (req, res) => {
  const { code } = req.params;
  const file = fileModel.findByCode(code);
  if (!file) {
    return res.status(404).json({ message: 'File not found' });
  }

  const filePath = path.join(uploadsDir, file.stored_name);
  if (!fs.existsSync(filePath)) {
    return res.status(410).json({ message: 'File no longer available' });
  }

  fileModel.incrementDownload(code);

  res.setHeader('X-Download-Code', code);
  res.download(filePath, file.name);
});

router.delete('/:code', (req, res) => {
  const { code } = req.params;
  const deleted = fileModel.deleteByCode(code, uploadsDir);
  if (!deleted) {
    return res.status(404).json({ message: 'File not found' });
  }
  return res.status(204).end();
});

router.post('/', (req, res, next) => {
  upload.single('file')(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ message: 'File exceeds the 10 GB limit.' });
      }
      return res.status(400).json({ message: err.message });
    }
    if (err) {
      return next(err);
    }

    const file = req.file;
    if (!file) {
      return res.status(400).json({ message: 'No file uploaded.' });
    }

    const checksum = crypto.createHash('sha1');
    const stream = fs.createReadStream(file.path);

    stream.on('data', (chunk) => checksum.update(chunk));

    stream.on('error', (streamErr) => {
      next(streamErr);
    });

    stream.on('end', () => {
      const digest = checksum.digest('hex');
      const code = generateShareCode();
      const saved = fileModel.insertFile({
        code,
        name: file.originalname,
        storedName: file.filename,
        size: file.size,
        mimeType: file.mimetype,
        checksum: digest
      });

      res.status(201).json({
        link: buildShareLink(req, code),
        file: formatFile(saved, req)
      });
    });
  });
});

module.exports = router;
