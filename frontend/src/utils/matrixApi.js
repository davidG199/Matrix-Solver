// Utilidades de cliente para validar y construir payloads de matrices

export function createMatrix(rows, cols, fill = "") {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => fill),
  );
}

export function resizeMatrix(matrix, rows, cols) {
  return Array.from({ length: rows }, (_, rowIndex) =>
    Array.from(
      { length: cols },
      (_, colIndex) => matrix?.[rowIndex]?.[colIndex] ?? "",
    ),
  );
}

export function clampInt(value, min, max) {
  const parsedValue = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsedValue)) return min;
  return Math.min(max, Math.max(min, parsedValue));
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function formatNumber(value) {
  if (!Number.isFinite(value)) return String(value);
  const rounded = Math.round(value);
  if (Math.abs(value - rounded) < 1e-10) return String(rounded);
  return Number(value.toFixed(6)).toString();
}

// Lee y valida una matriz (cliente): convierte celdas a Number y lanza Error si no son válidas
export function readMatrix(matrix, label) {
  return matrix.map((row, rowIndex) =>
    row.map((value, colIndex) => {
      const trimmed = String(value).trim();
      if (trimmed === "") {
        throw new Error(
          `${label} fila ${rowIndex + 1} columna ${colIndex + 1}: valor vacío`,
        );
      }

      const number = Number(trimmed);
      if (!Number.isFinite(number)) {
        throw new Error(
          `${label} fila ${rowIndex + 1} columna ${colIndex + 1}: no es un número válido`,
        );
      }

      return number;
    }),
  );
}

export function getMatrixShape(matrix) {
  const rows = matrix.length;
  const cols = rows > 0 ? matrix[0].length : 0;
  return { rows, cols };
}

// Valida dimensiones según la operación (cliente)
export function validateOperationDimensions(operation, matrixA, matrixB) {
  const shapeA = getMatrixShape(matrixA);
  const shapeB = getMatrixShape(matrixB);

  if (operation.key === "suma" || operation.key === "resta") {
    if (shapeA.rows !== shapeB.rows || shapeA.cols !== shapeB.cols) {
      throw new Error(
        `Dimensiones incompatibles: A ${shapeA.rows}x${shapeA.cols}, B ${shapeB.rows}x${shapeB.cols}. Para ${operation.label} ambas deben coincidir.`,
      );
    }
  }

  if (operation.key === "multiplicacion") {
    if (shapeA.cols !== shapeB.rows) {
      throw new Error(
        `Dimensiones incompatibles para multiplicación: columnas de A (${shapeA.cols}) deben igualar filas de B (${shapeB.rows}).`,
      );
    }
  }

  //validar que la matriz sea cuadrada
  if (operation.kind === "single") {
    if (shapeA.rows !== shapeA.cols) {
      throw new Error(
        `${operation.label}: la matriz debe ser cuadrada (${shapeA.rows}x${shapeA.cols}).`,
      );
    }
  }
}

// Construye el payload JSON para enviar al backend (cliente)
export function buildRequestPayload(operationKey, state, operationMap) {
  const operation = operationMap[operationKey];
  const matrices = state.matrices[operationKey];

  const matrixA = readMatrix(matrices.a, "Matriz A");
  const matrixB = readMatrix(matrices.b, "Matriz B");

  validateOperationDimensions(operation, matrixA, matrixB);

  return {
    matrix_a: matrixA,
    matrix_b: matrixB,
  };
}
