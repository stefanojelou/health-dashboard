# Documentación del esquema `identity`

Base de datos MySQL que soporta los servicios de verificación de identidad (KYC) y firma electrónica de Jelou. Utiliza Prisma como ORM.

---

## Resumen de arquitectura

El esquema gira en torno a dos flujos principales que operan de forma independiente:

1. **Verificación biométrica (KYC)**: Un cliente (`clients`) inicia una sesión de verificación (`biometrics`) que se descompone en pasos (`biometric_steps`): verificación documental, prueba de vida y comparación facial. Cada paso tiene tablas de detalle dedicadas.

2. **Firma electrónica**: Un cliente firma documentos a través de proveedores externos como FAD. Este flujo se rastrea en `signatures` y no está vinculado a la cadena biométrica — se conecta al cliente y a la ejecución de Brain de forma independiente.

Ambos flujos se originan desde ejecuciones de workflows de Brain (skill_id, execution_id, node_id en los JSON de contexto) y están asociados a empresas del ecosistema Jelou a través de `companyId`.

---

## Modelo de entidades

### `companies`

Empresas cliente de Jelou que utilizan los servicios de identidad. El `companyId` es numérico y coincide con el ID de empresa del esquema `chatbot` — este es el campo de unión principal con el resto de la plataforma.

> **Nota**: No se ha inspeccionado el contenido de esta tabla en detalle. Probablemente contiene nombre, configuración y estado de cada empresa.

### `company_settings`

Configuración específica por empresa para los servicios de identidad: umbrales de aprobación, proveedores habilitados, reglas de negocio, etc.

> **Nota**: No se ha inspeccionado. Podría contener los umbrales de similitud para facematch, configuración de human-in-the-loop, y proveedores activos.

### `clients`

El usuario final que está siendo verificado. Representa a la persona física que pasa por el proceso de KYC o firma un documento.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID del cliente |
| `names` | varchar | Nombres del cliente |
| `surnames` | varchar | Apellidos (puede ser NULL) |
| `tokenId` | varchar | Identificador de sesión, generalmente el número de teléfono |
| `phone` | varchar | Número de teléfono (formato internacional, ej: `593963546336`) |
| `email` | varchar | Email (generalmente NULL — la verificación es vía WhatsApp) |
| `isBlacklisted` | tinyint | Flag de lista negra para control de fraude |
| `blockedUntil` | datetime | Bloqueo temporal del cliente |
| `lastBiometricId` | varchar | FK desnormalizada a su verificación biométrica más reciente |
| `password` | varchar | Generalmente NULL |
| `createdAt` / `updatedAt` | datetime | Timestamps |

**Notas importantes**:
- No tiene `companyId`. La relación cliente → empresa se establece a través de `biometrics`. Un mismo cliente puede ser verificado por múltiples empresas.
- El teléfono es el identificador principal del cliente en la práctica.

### `users`

Usuarios internos del sistema de identidad (operadores, supervisores, administradores). No confundir con `clients`.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `email` | varchar(191) | Email del usuario |
| `password` | varchar(191) | Contraseña hasheada |
| `firstName` / `lastName` | varchar(100) | Nombre del operador |
| `status` | enum | `Active`, `Inactive`, `Blocked` |
| `role` | enum | `Admin`, `Operator`, `Supervisor`, `Viewer` |
| `lastLoginAt` | datetime | Último acceso |
| `deletedAt` | datetime | Soft delete |

**Uso**: Estos son los usuarios que revisan verificaciones en modo human-in-the-loop, asignan estados, y gestionan el backoffice de identidad.

---

## Flujo de verificación biométrica (KYC)

### `biometrics`

**Tabla central del flujo KYC.** Cada fila representa una sesión de verificación de identidad completa para un cliente, iniciada desde un workflow de Brain.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID de la sesión |
| `status` | varchar | Estado general: `Initiated`, `Incomplete`, `Approved`, `Disapproved`, `Failed`, `Cancelled`, `Abandoned`, `Review` |
| `details` | JSON | Contexto de ejecución: `bot_id`, `user_id`, `skill_id`, `execution_id`, `node_id`. En sesiones completadas también incluye `result_facematch` con score y URLs |
| `clientId` | varchar(191) | FK → `clients.id` |
| `channelId` | varchar(191) | FK → `channels.id` |
| `companyId` | int | ID de empresa (numérico, coincide con `chatbot.companies.id`) |
| `isVerified` | tinyint | Flag booleano de verificación |
| `verifiedBy` | varchar | Usuario que verificó manualmente |
| `verificationStatus` | varchar | Estado de verificación manual |
| `verificationComments` | varchar | Comentarios del revisor |
| `expiredAt` | datetime | Expiración de la sesión |
| `createdAt` / `updatedAt` | datetime | Timestamps |

**Queries típicas**:
- Volumen de KYC por empresa y mes
- Tasa de aprobación general
- Sesiones iniciadas pero no completadas (funnel drop-off)

```sql
-- Volumen mensual por estado
SELECT
  DATE_FORMAT(createdAt, '%Y-%m') AS mes,
  status,
  COUNT(*) AS cantidad
FROM identity.biometrics
WHERE createdAt >= '2026-01-01'
GROUP BY mes, status
ORDER BY mes, status;
```

### `biometric_steps`

**Detalle paso a paso de cada sesión biométrica.** Una sesión de `biometrics` se descompone en múltiples pasos, cada uno representando una etapa del KYC.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID del paso |
| `stepName` | varchar(255) | Tipo de paso: `document_check`, `liveness`, `facematch` |
| `description` | varchar(255) | Descripción textual (ej: "Facematch verification step", "Se detecto fraude LipSync v3") |
| `status` | enum | `Approved`, `Disapproved`, `Initiated`, `Incomplete`, `Cancelled`, `Abandoned`, `Failed`, `Review`, `Rejected`, `Fraud` |
| `data` | JSON | **Payload pesado.** Contiene resultados completos según el tipo de paso (ver abajo) |
| `details` | JSON | Contexto de Brain: `node_id`, `skill_id`, `execution_id` |
| `biometricId` | varchar(191) | FK → `biometrics.id` |
| `enableHumanInLoop` | tinyint | Si el paso requiere revisión humana |
| `assignedTo` | varchar | UUID del operador asignado para revisión |
| `updatedBy` | varchar | Quién actualizó el estado |
| `updateComment` | varchar | Comentario de la actualización |
| `documentCheckId` | varchar(191) | FK → `document_check_steps.id` (solo si stepName = document_check) |
| `facematchId` | varchar(191) | FK → `facematch_steps.id` (solo si stepName = facematch) |
| `livenessId` | varchar(191) | FK → `liveness_steps.id` (solo si stepName = liveness) |

**Contenido del campo `data` según stepName**:

- **`document_check`**: Resultado completo de OCR (Regula) y validación gubernamental. Incluye `document_check_data.verified_fields` (todos los campos extraídos del documento: nombre, cédula, fecha de nacimiento, etc.), `status_fields` (resultado óptico), `gov_entity_fields` (validación contra registro civil), `images_extracted` (retrato, firma, código de barras), e `image_quality_details`.
- **`liveness`**: OTP de prueba de vida (`otp`: "cuatro-seis-uno-cinco"), URL del video, y URL del selfie extraído.
- **`facematch`**: Generalmente vacío en esta tabla — el detalle vive en `facematch_steps`.

**Query principal — KYC por etapa y mes**:

```sql
SELECT
  DATE_FORMAT(bs.createdAt, '%Y-%m') AS mes,
  bs.stepName,
  bs.status,
  COUNT(*) AS cantidad
FROM identity.biometric_steps bs
WHERE bs.createdAt >= '2026-01-01'
GROUP BY mes, bs.stepName, bs.status
ORDER BY mes, bs.stepName, bs.status;
```

### `document_check_steps`

Detalle relacional de la verificación documental. Versión normalizada de lo que también existe como JSON en `biometric_steps.data`.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `documentFrontalUrl` | varchar | Imagen del frente del documento |
| `documentBackUrl` | varchar | Imagen del reverso |
| `entityResponse` | tinyint | **Resultado de validación gubernamental** (1 = válido, 0 = no válido o no disponible) |
| `regulaResponse` | tinyint | Resultado del OCR de Regula (1 = exitoso) |
| `documentIdentityId` | varchar(191) | FK → `document_identities.id` |

**Uso clave**: El campo `entityResponse` es la métrica de "Validación gubernamental" — indica si el documento fue verificado contra la base de datos del registro civil del país correspondiente.

```sql
-- Validación gubernamental por mes
SELECT
  DATE_FORMAT(bs.createdAt, '%Y-%m') AS mes,
  dcs.entityResponse,
  COUNT(*) AS cantidad
FROM identity.biometric_steps bs
JOIN identity.document_check_steps dcs ON dcs.id = bs.documentCheckId
WHERE bs.createdAt >= '2026-01-01'
  AND bs.stepName = 'document_check'
GROUP BY mes, dcs.entityResponse
ORDER BY mes;
```

### `document_identities`

Datos de identidad extraídos del documento verificado. Cada fila contiene la información personal parseada por OCR.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `clientId` | varchar(191) | FK → `clients.id` |
| `documentNumber` | varchar(100) | Número de cédula/DNI |
| `givenNames` / `surname` | varchar(255) | Nombres y apellidos extraídos |
| `birthDate` | datetime | Fecha de nacimiento |
| `age` | int | Edad calculada |
| `sex` | varchar(20) | Sexo |
| `nationality` | varchar(100) | Nacionalidad |
| `country` | varchar(100) | País emisor |
| `documentType` | varchar(100) | Tipo de documento (identity_card, passport, etc.) |
| `dateOfExpiry` | datetime | Fecha de vencimiento |
| `fingerprintCode` | varchar(100) | Código dactilar (para documentos ecuatorianos) |

### `facematch_steps`

Resultado de la comparación facial entre el selfie de prueba de vida y la foto del documento.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `similarity` | int | Score de similitud (0-100). En los datos observados, todos los aprobados tienen 99 |
| `imageLivenessUrl` | varchar | URL del selfie de prueba de vida |
| `imageDocumentUrl` | varchar | URL del retrato extraído del documento |

**Nota**: Algunas filas tienen URLs NULL — en esos casos las imágenes están almacenadas únicamente en el JSON de `biometric_steps.data` o en el JSON de `biometrics.details`.

### `liveness_steps`

Registro de la prueba de vida. Tabla slim con solo la evidencia del challenge.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `mediaUrl` | varchar | URL del video de prueba de vida (almacenado en CDN de Jelou) |
| `otpCode` | varchar | Código OTP que el usuario debió pronunciar |

**Nota sobre formatos de OTP**: Se observan al menos dos modalidades de challenge:
- Dígitos en palabras: `"cuatro-seis-uno-cinco"`
- Fechas completas: `"12 de diciembre del 2025"`

Esto sugiere que existen diferentes configuraciones de prueba de vida según empresa o nivel de seguridad.

---

## Firma electrónica

### `signatures`

Registro de solicitudes de firma electrónica enviadas a proveedores externos.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `clientId` | varchar(191) | FK → `clients.id` |
| `biometricId` | varchar(191) | FK → `biometrics.id` (generalmente NULL — firmas y biometría son flujos paralelos) |
| `companyId` | varchar(191) | ID de empresa (numérico como string) |
| `executionId` | varchar(191) | ID de ejecución de Brain |
| `providerRequestId` | varchar(191) | ID de la solicitud en el proveedor externo |
| `provider` | varchar(191) | Proveedor de firma: `FAD` (Firma Autógrafa Digital) |
| `status` | enum | `Pending`, `Sent`, `Processing`, `Signed`, `Completed`, `Rejected`, `Expired`, `Failed`, `Cancelled` |
| `providerResponse` | JSON | Respuesta del proveedor con detalles de la firma (device, geolocation, archivos generados) |
| `metadata` | JSON | Contexto de Brain: `bot_id`, `user_id`, `company_name`, `document_type`, `jelou_notification` |
| `createdAt` | datetime | Fecha de creación |
| `completedAt` | datetime | Fecha de finalización (NULL si no se ha completado) |

**Estado actual**: Solo se observan datos de enero 2026, todos de companyId 1944 ("Jelou Marketplace"). Esto indica que la funcionalidad de firma electrónica está en fase de desarrollo/testing interno y aún no se ha desplegado a clientes en producción.

```sql
-- Firmas por mes, proveedor y estado
SELECT
  DATE_FORMAT(s.createdAt, '%Y-%m') AS mes,
  s.provider,
  s.status,
  COUNT(*) AS cantidad
FROM identity.signatures s
WHERE s.createdAt >= '2026-01-01'
GROUP BY mes, s.provider, s.status
ORDER BY mes, s.provider, s.status;
```

---

## Tablas de soporte

### `channels`

Canal de comunicación por el cual se inicia la verificación. Referenciado desde `biometrics.channelId`.

> **Nota**: No inspeccionada en detalle. Probablemente contiene el tipo de canal (WhatsApp, web, etc.) y la configuración del bot asociado.

### `tokens`

Tokens de sesión para el proceso de verificación.

> **Nota**: No inspeccionada. Probablemente maneja la autenticación de sesiones de verificación.

### `collections`

Agrupaciones o lotes de verificaciones.

> **Nota**: No inspeccionada. Podría ser un mecanismo para agrupar verificaciones por campaña, lote, o proceso de negocio.

### `authentications`

Registros de autenticación.

> **Nota**: No inspeccionada. Podría estar relacionada con autenticaciones de API o de usuarios del backoffice.

### `provider_api_keys`

Claves de API de los proveedores de verificación (Regula, registros civiles, FAD, etc.) por empresa.

> **Nota**: No inspeccionada. Contendría las credenciales encriptadas para cada proveedor.

### `security_events`

Eventos de seguridad detectados durante el proceso de verificación.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `type` | varchar(100) | Tipo de evento. Observado: `DEVICE_MISMATCH` |
| `companyId` | varchar(191) | Empresa donde ocurrió |
| `sessionId` / `tokenId` | varchar | Identificadores de sesión |
| `expected` | JSON | Huella digital del dispositivo esperado |
| `actual` | JSON | Huella digital del dispositivo actual |
| `severity` | varchar(50) | Severidad: `HIGH` |
| `resolved` | tinyint | Si fue resuelto (0 = no) |

**Estado actual**: Todos los registros observados son `DEVICE_MISMATCH` de companyId 1944, lo que sugiere que esta funcionalidad está en pruebas. El JSON contiene datos ricos de fingerprinting: user agent, IP, geolocalización, resolución de pantalla, timezone.

### `tools`

Registro de herramientas/servicios externos invocados durante la verificación.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | varchar(191) PK | UUID |
| `biometricId` | varchar(191) | FK → `biometrics.id` |
| `toolName` | varchar(191) | Nombre de la herramienta invocada |
| `status` | varchar(191) | Estado de la ejecución |
| `encryptedRequest` / `encryptedResponse` | longblob | Request/response encriptados |
| `ivRequest` / `ivResponse` | longblob | Vectores de inicialización para descifrado |
| `authenticationId` | varchar(191) | FK → `authentications.id` |
| `executedAt` / `createdAt` | datetime | Timestamps |

**Nota**: Los payloads están encriptados (AES con IV), lo que indica que esta tabla maneja datos sensibles de proveedores.

### `params`

Tabla de configuración de parámetros.

> **Nota**: No inspeccionada.

### `client_block_history`

Historial de bloqueos de clientes. Complementa los campos `isBlacklisted` y `blockedUntil` de `clients`.

> **Nota**: No inspeccionada.

### `audit_summary_info` / `summary_biometrics` / `summary_tools`

Tablas de resumen pre-computadas, probablemente para dashboards o reportes.

> **Nota**: No inspeccionadas. Podrían ser útiles para queries de alto nivel sin necesidad de agregar desde las tablas transaccionales.

### `_prisma_migrations`

Tabla interna de Prisma ORM para control de migraciones de esquema. No relevante para análisis.

---

## Relaciones clave

```
companies (companyId numérico)
  └── biometrics (sesión KYC)
        ├── biometric_steps (pasos)
        │     ├── document_check_steps (detalle documental)
        │     │     └── document_identities (datos extraídos)
        │     ├── liveness_steps (prueba de vida)
        │     └── facematch_steps (comparación facial)
        └── clients (persona verificada)
              └── signatures (firma electrónica, flujo paralelo)
```

## Joins útiles

**Biometrics → Company (cruce con chatbot schema)**:
`identity.biometrics.companyId = chatbot.companies.id`

**Biometric step → Detalle de documento**:
`identity.biometric_steps.documentCheckId = identity.document_check_steps.id`

**Biometric step → Detalle de liveness**:
`identity.biometric_steps.livenessId = identity.liveness_steps.id`

**Biometric step → Detalle de facematch**:
`identity.biometric_steps.facematchId = identity.facematch_steps.id`

**Document check → Identidad extraída**:
`identity.document_check_steps.documentIdentityId = identity.document_identities.id`

**Contexto de Brain (en JSON)**:
Los campos `biometrics.details` y `biometric_steps.details` contienen `skill_id` y `execution_id` que pueden cruzarse con `workflow_executions` en MongoDB para obtener el contexto completo del flujo conversacional.

---

## Métricas principales que soporta este esquema

| Métrica | Tabla(s) | Campo(s) clave |
|---------|----------|----------------|
| KYC por etapa (funnel) | `biometric_steps` | `stepName`, `status`, `createdAt` |
| Firmas electrónicas | `signatures` | `status`, `provider`, `createdAt` |
| Validación gubernamental | `document_check_steps` | `entityResponse` |
| Tasa de fraude (liveness) | `biometric_steps` | `status = 'Fraud'` o `description LIKE '%fraude%'` |
| Score de facematch | `facematch_steps` | `similarity` |
| Países verificados | `document_identities` | `country`, `nationality` |
| Eventos de seguridad | `security_events` | `type`, `severity` |
| Volumen por empresa | `biometrics` | `companyId`, `createdAt` |
