const { customAlphabet } = require('nanoid');

const alphabet = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
const nanoid = customAlphabet(alphabet, 8);

function generateShareCode() {
  return nanoid();
}

module.exports = {
  generateShareCode
};
