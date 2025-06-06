const debug = require("debug");
const config = require("../config");

const debugLog = debug(config.debug);

const logger = {
  info: (message) => console.log(`[INFO] ${message}`),
  error: (message) => console.error(`[ERROR] ${message}`),
  warn: (message) => console.warn(`[WARN] ${message}`),
  debug: (message) => debugLog(message),
};

module.exports = logger;
