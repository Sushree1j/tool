const { generateShareCode } = require('../utils/generateCode');

describe('generateShareCode', () => {
  it('returns an 8-character code', () => {
    const code = generateShareCode();
    expect(code).toHaveLength(8);
  });

  it('avoids ambiguous characters', () => {
    const alphabet = new Set('ABCDEFGHJKMNPQRSTUVWXYZ23456789'.split(''));
    for (let i = 0; i < 20; i += 1) {
      const code = generateShareCode();
      code.split('').forEach((char) => {
        expect(alphabet.has(char)).toBe(true);
      });
    }
  });
});
