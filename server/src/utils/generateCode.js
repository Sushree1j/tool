const crypto = require('crypto');

const alphabet = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
const alphabetLength = alphabet.length;

function generateShareCode(length = 8) {
  const randomBytes = crypto.randomBytes(length);
  let code = '';
  for (let i = 0; i < length; i += 1) {
    const index = randomBytes[i] % alphabetLength;
    code += alphabet[index];
  }
  return code;
}

module.exports = {
  generateShareCode
};
