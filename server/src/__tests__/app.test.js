const path = require('path');
const fs = require('fs');
const request = require('supertest');

process.env.NODE_ENV = 'test';
process.env.CLIENT_ORIGIN = 'http://localhost:5173';

const app = require('../app');
const { closeDb } = require('../db');

const uploadsDir = path.join(__dirname, '..', 'uploads');

describe('App', () => {
  afterAll(() => {
    if (fs.existsSync(uploadsDir)) {
      fs.rmSync(uploadsDir, { recursive: true, force: true });
    }
    closeDb();
  });

  it('responds to /health', async () => {
    const res = await request(app).get('/health');
    expect(res.status).toBe(200);
    expect(res.body.status).toEqual('ok');
  });

  it('uploads a file and returns metadata', async () => {
    const response = await request(app)
      .post('/api/files')
      .attach('file', Buffer.from('NovaShare test file'), 'sample.txt');

    expect(response.status).toBe(201);
    expect(response.body.file).toMatchObject({
      name: 'sample.txt'
    });
    expect(response.body.link).toContain('/api/files/');

    const { code } = response.body.file;
    expect(code).toHaveLength(8);

    await request(app).delete(`/api/files/${code}`);
  });
});
