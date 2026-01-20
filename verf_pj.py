from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from functools import reduce

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

fech_par = '20250630'

spark = SparkSession.builder \
    .appName("verf_campos_pj") \
    .config("spark.sql.session.locale", "es") \
    .getOrCreate()

spark.conf.set("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "LEGACY")
spark.conf.set("spark.sql.legacy.parquet.int96RebaseModeInRead", "LEGACY")
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
spark.conf.set("spark.sql.parquet.enableVectorizedReader", "false")
spark.conf.set("spark.sql.files.ignoreCorruptFiles", "true")

df_tip_pj = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("delimiter", ";") \
    .load("/user/T45109/AD/VerfCamposPJ/ANEXO2_PERSONA_JURIDICA_20251130.csv")

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Tipos de validación
TIPO_NULL = "NULL"
TIPO_VACIO = "VACIO"
TIPO_OBLIGADO = "CAMP_OBLIGADO"
TIPO_CONDICIONAL = "CAMP_CONDICIONAL"
TIPO_OPC_CONDICIONAL = "CAMP_OPC_CONDICIONAL"
TIPO_OPCIONAL = "CAMP_OPCIONAL"
TIPO_INCONSISTENCIA = "INCONSISTENCIA_DATOS"

# Severidades
SEV_ALTA = "ALTA"
SEV_MEDIA = "MEDIA"
SEV_BAJA = "BAJA"

# Regex para palabras inválidas
REGEX_PALABRA_INVALIDA = (
    r"(?i)\b("
    r"no tiene|no especifica|no existe|no contiene|sin informacion|sin información|"
    r"desconocido|n/a|na|null|ninguno|sin dato|dato no disponible|no aplica|"
    r"no corresponde|no registrado|no declarado|no proveído|no proporcionado|"
    r"información no disponible|información desconocida|s/d|s/n|sn"
    r")\b"
)

# Patrones de teclado sospechosos
KEYBOARD_BAD = ["QWERTY", "ASDFGH", "ZXCVBN", "123456", "ABCDEF"]
EXC_2L = {"SA", "CV", "RL", "SL", "SC"}

# Regex para secuencias ascendentes/descendentes
_ASC_REGEX = r"(0123|1234|2345|3456|4567|5678|6789|7890|8901|9012)"
_DESC_REGEX = r"(9876|8765|7654|6543|5432|4321|3210|2109|1098|0987)"

# Schema para observaciones
OBS_SCHEMA = ArrayType(
    StructType([
        StructField("idx", IntegerType(), False),
        StructField("campo", StringType(), False),
        StructField("motivo", StringType(), False),
        StructField("tipo", StringType(), False),
        StructField("severidad", StringType(), False)
    ])
)

# ============================================================================
# DEFINICIÓN DE CAMPOS POR CATEGORÍA
# ============================================================================

ALL_FIELDS = df_tip_pj.columns
field_to_idx = {c: i for i, c in enumerate(ALL_FIELDS)}

# Campos OBLIGATORIOS (9 campos)
CAMP_OBLIGADOS = [
    "CODIGO",
    "TIPO_SOCIEDAD_COMERCIAL",
    "CODIGO_ACTIVIDAD_ECONOMICA",
    "DESCRIPCION_ACTIVIDAD_ECONOMICA",
    "NIT",
    "REFERENCIAS_COMERCIALES",
    "PAF_ALTA",
    "FECHA_ALTA",
    "OPERACION_ASOCIADA_ALTA"
]

# Campos CONDICIONALES (21 campos) - Ver reglas más abajo
CAMP_CONDICIONALES = [
    "RAZON_SOCIAL",
    "IDENTIFICACION_CANT_INT_ALTA_GERENCIA",
    "CARGO_ALTA_GERENCIA",
    "NOMBRES_APELLIDOS_CARGO_GERENCIA",
    "TIPO_DOC_ID",
    "NRO_DOC_ID",
    "DOMICILIO",
    "NOMBRE_REP_LEGAL",
    "NACIONALIDAD_REP_LEGAL",
    "PAIS_RESIDENCIA_REP_LEGAL",
    "FECHA_NACIMIENTO_REP_LEGAL",
    "ESTADO_CIVIL_REP_LEGAL",
    "NOMBRE_CONYUGE_REP_LEGAL",
    "DOMICILIO_PARTICULAR_REP_LEGAL",
    "PROFESION_REP_LEGAL",
    "CODIGO_ACTIVIDAD_ECO_REP_LEGAL",
    "DESC_ACTIVIDAD_ECO_REP_LEGAL",
    "LUGAR_TRABAJO_REP_LEGAL",
    "CARGO_REP_LEGAL",
    "NIVEL_INGRESOS_REP_LEGAL",
    "NIVEL_INGRESOS_REP_LEGAL_SUS",
    "FECHA_INGRESO_TRAB_REP_LEGAL"
]

# Campos OPCIONALES CONDICIONALES (16 campos)
CAMP_OPC_CONDICIONALES = [
    "FECHA_ULTIMA_TRANSACCION",
    "TIPO_CUENTA_ULTIMA_TRANSACCION",
    "CANAL_ATENCION_ULT_TRANS",
    "TELEFONO",
    "CODIGO_REP_LEGAL",
    "TIPO_DOCUMENTO_IDENT_REP_LEGAL",
    "DOCUMENTO_IDENTIDAD_REP_LEGAL",
    "TELEFONO_REP_LEGAL",
    "CORREO_ELECTRONICO_REP_LEGAL",
    "FECHA_INGRESO_TRABAJO_REP_LEGAL",
    "MONEDA_NIVEL_INGRESOS_REP_LEGAL",
    "REFERENCIAS_PERS_REP_LEGAL",
    "REFERENCIAS_COM_REP_LEGAL",
    "REFERENCIAS_BANCARIAS_REP_LEGAL",
    "TIPO_RIESGO_REP_LEGAL",
    "FECHA_ULTIMA_ACTUALIZACION",
    "CANTIDAD_ACTUALIZACIONES"
]

# Campos OPCIONALES (resto de campos)
CAMP_OPCIONALES = [
    "EXT_DOC_ID",
    "NUMERO_REGISTRO",
    "CLASIFICACION_TIPO_RIESGO",
    "EXP_DOCUMENTO_IDENTIDAD_REP_LEGAL",
    "NIT_REP_LEGAL",
    "DOMICILIO_COMERCIAL_REP_LEGAL",
    "CORREO_ELECTRONICO_REP_LEGAL",
    "REFERENCIAS_PERS_REP_LEGAL",
    "REFERENCIAS_COM_REP_LEGAL",
    "REFERENCIAS_BANCARIAS_REP_LEGAL"
]

# ============================================================================
# FUNCIONES DE VALIDACIÓN BASE
# ============================================================================

def is_null(c):
    """Verifica si una columna es NULL."""
    return col(c).isNull()

def is_vacio(c):
    """Verifica si una columna es vacía (string sin contenido después de trim)."""
    return (col(c).isNotNull()) & (trim(col(c)) == "")

def is_empty(c):
    """Verifica si una columna es NULL o vacía."""
    return is_null(c) | is_vacio(c)

def campo_ok(c):
    """Verifica si un campo tiene datos válidos (no null, no vacío)."""
    return ~is_empty(c)

def upper_trim(c):
    """Retorna columna en mayúsculas y sin espacios."""
    return upper(trim(col(c)))

def only_digits_str(c):
    """Extrae solo los dígitos de una columna."""
    return regexp_replace(col(c), r'[^0-9]', '')

def numeric_str_to_double(c):
    """Convierte string numérico a double, manejando comas/puntos."""
    cleaned = regexp_replace(col(c), r'[^\d.,\-]', '')
    normalized = regexp_replace(cleaned, ',', '.')
    return normalized.cast("double")

# ============================================================================
# FUNCIONES DE VALIDACIÓN DE INCONSISTENCIAS
# ============================================================================

def tiene_palabra_invalida(c):
    """Detecta palabras placeholder/inválidas según REGEX_PALABRA_INVALIDA."""
    return campo_ok(c) & col(c).rlike(REGEX_PALABRA_INVALIDA)

def email_invalido(c):
    """Valida formato de email solo si el campo tiene contenido."""
    return when(is_empty(c), lit(False)).otherwise(
        ~col(c).rlike(r"^[A-Za-z0-9._%+\-]{3,}@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$") |
        col(c).rlike(r"(?i)(.)\1{3,}") |   # 4 caracteres repetidos
        col(c).rlike(r"^[0-9]+@")          # usuario solo dígitos
    )

def phone_invalido(c):
    """Valida teléfono (7-8 dígitos, sin repeticiones)."""
    digits = only_digits_str(c)
    all_same = digits.rlike(r"^([0-9])\1+$")
    return when(is_empty(c), lit(False)).otherwise(
        (length(digits) < 7) | (length(digits) > 8) | all_same
    )

def nit_formato_invalido(c):
    """Valida formato NIT (mínimo 8 dígitos, termina en 01-04)."""
    d = only_digits_str(c)
    ends_ok = d.rlike(r"(01|02|03|04)[1-9]$")
    only_digits_same_len = (length(d) == length(regexp_replace(col(c), r'\s+', '')))
    return campo_ok(c) & ((length(d) < 8) | (~ends_ok) | (~only_digits_same_len))

def domicilio_invalido(c, min_len=5):
    """Valida domicilio (mínimo longitud, no solo números, sin placeholders)."""
    s = trim(col(c))
    tiene = campo_ok(c)
    only_digits = length(regexp_replace(s, r'[0-9]', '')) == 0
    return tiene & ((length(s) < min_len) | only_digits | col(c).rlike(REGEX_PALABRA_INVALIDA))

def text_invalido(c, min_len=3):
    """
    Validación textual completa:
    - Palabras inválidas (placeholders)
    - Demasiado corto
    - Solo dígitos
    - Repeticiones de caracteres/secuencias
    - Sin vocales (para textos >= 3 letras)
    - Patrones de teclado
    """
    s = upper_trim(c)
    tiene = campo_ok(c)

    too_short = (length(trim(col(c))) < min_len)
    only_digits = (length(regexp_replace(col(c), r'[0-9]', '')) == 0)
    rep_char = s.rlike(r"(.)\1{3,}")
    rep_seq = s.rlike(r"^([A-Z0-9]{1,3})\1{2,}$")
    
    letters = regexp_replace(s, r'[^A-ZÁÉÍÓÚÜÑ]', '')
    no_vowels = (length(letters) >= 3) & (~letters.rlike(r"[AEIOUÁÉÍÓÚÜ]"))
    
    any_kb = lit(False)
    for kb in KEYBOARD_BAD:
        any_kb = any_kb | s.contains(kb)
    
    bad_word = col(c).rlike(REGEX_PALABRA_INVALIDA)
    two_letter_exc = (length(letters) == 2) & letters.isin(list(EXC_2L))
    
    return tiene & (too_short | only_digits | rep_char | rep_seq | any_kb | (no_vowels & ~two_letter_exc) | bad_word)

def numeric_seq_or_repeat(c, min_len=4):
    """Detecta secuencias o repeticiones numéricas (1111, 1234, 9876, etc)."""
    d = only_digits_str(c)
    len_ok = length(d) >= min_len
    repetido = d.rlike(r"^(\d)\1{" + str(min_len-1) + r",}$")
    asc = d.rlike(_ASC_REGEX)
    desc = d.rlike(_DESC_REGEX)
    return campo_ok(c) & len_ok & (repetido | asc | desc)

def letras_repetidas(c, min_rep=4):
    """Detecta letras repetidas consecutivas (AAAA, ZZZZ, etc)."""
    s = upper_trim(c)
    return campo_ok(c) & s.rlike(r"([A-ZÁÉÍÓÚÜÑ])\1{" + str(min_rep-1) + r",}")

def is_pos_number(c):
    """Verifica si es un número positivo."""
    return campo_ok(c) & (col(c).cast("int") > 0)

def is_nonneg_number(c):
    """Verifica si es un número no negativo."""
    return campo_ok(c) & numeric_str_to_double(c).isNotNull() & (numeric_str_to_double(c) >= 0.0)

# ============================================================================
# FUNCIÓN PRINCIPAL DE OBSERVACIONES
# ============================================================================

def add_obs(df, condicion, campo, motivo, tipo, severidad):
    """Agrega una observación al array OBS_ARRAY cuando se cumple la condición."""
    return df.withColumn(
        "OBS_ARRAY",
        when(
            condicion,
            concat(
                col("OBS_ARRAY"),
                array(
                    struct(
                        lit(field_to_idx[campo]).alias("idx"),
                        lit(campo).alias("campo"),
                        lit(motivo).alias("motivo"),
                        lit(tipo).alias("tipo"),
                        lit(severidad).alias("severidad")
                    )
                )
            )
        ).otherwise(col("OBS_ARRAY"))
    )

# ============================================================================
# VALIDACIÓN PASO 1: NULL Y VACÍO
# ============================================================================

print("=== PASO 1: Validando NULL y VACÍO ===")
df = df_tip_pj.withColumn("OBS_ARRAY", array().cast(OBS_SCHEMA))

for c in ALL_FIELDS:
    df = add_obs(df, is_null(c), c, f"{c}_NULL", TIPO_NULL, SEV_ALTA)
    df = add_obs(df, is_vacio(c), c, f"{c}_VACIO", TIPO_VACIO, SEV_ALTA)

# ============================================================================
# VALIDACIÓN PASO 2: CAMPOS OBLIGATORIOS
# ============================================================================

print("=== PASO 2: Validando CAMPOS OBLIGATORIOS ===")

for c in CAMP_OBLIGADOS:
    # Si está vacío
    df = add_obs(
        df,
        is_empty(c),
        c,
        f"{c}_FALTANTE",
        TIPO_OBLIGADO,
        SEV_ALTA
    )
    
    # Si tiene palabra inválida (solo si no está vacío)
    df = add_obs(
        df,
        tiene_palabra_invalida(c),
        c,
        f"{c}_PALABRA_INVALIDA",
        TIPO_OBLIGADO,
        SEV_ALTA
    )

# Validaciones específicas para campos obligatorios
df = add_obs(
    df,
    campo_ok("NIT") & nit_formato_invalido("NIT"),
    "NIT",
    "NIT_FORMATO_INVALIDO",
    TIPO_OBLIGADO,
    SEV_ALTA
)

df = add_obs(
    df,
    campo_ok("RAZON_SOCIAL") & text_invalido("RAZON_SOCIAL", 3),
    "RAZON_SOCIAL",
    "RAZON_SOCIAL_TEXTO_INVALIDO",
    TIPO_OBLIGADO,
    SEV_ALTA
)

# ============================================================================
# VALIDACIÓN PASO 3: CAMPOS CONDICIONALES
# ============================================================================

print("=== PASO 3: Validando CAMPOS CONDICIONALES ===")

# REGLA: IDENTIFICACION_CANT_INT_ALTA_GERENCIA depende de campos 9 y 10
df = add_obs(
    df,
    campo_ok("IDENTIFICACION_CANT_INT_ALTA_GERENCIA") & 
    is_pos_number("IDENTIFICACION_CANT_INT_ALTA_GERENCIA") &
    is_empty("CARGO_ALTA_GERENCIA"),
    "IDENTIFICACION_CANT_INT_ALTA_GERENCIA",
    "IDENTIFICACION_CANT_INT_ALTA_GERENCIA_SIN_CARGO_ALTA_GERENCIA",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

df = add_obs(
    df,
    campo_ok("IDENTIFICACION_CANT_INT_ALTA_GERENCIA") & 
    is_pos_number("IDENTIFICACION_CANT_INT_ALTA_GERENCIA") &
    is_empty("NOMBRES_APELLIDOS_CARGO_GERENCIA"),
    "IDENTIFICACION_CANT_INT_ALTA_GERENCIA",
    "IDENTIFICACION_CANT_INT_ALTA_GERENCIA_SIN_NOMBRES_APELLIDOS",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

# REGLA: CARGO_ALTA_GERENCIA depende de campos 8 y 10
df = add_obs(
    df,
    campo_ok("CARGO_ALTA_GERENCIA") &
    (is_empty("IDENTIFICACION_CANT_INT_ALTA_GERENCIA") | 
     ~is_pos_number("IDENTIFICACION_CANT_INT_ALTA_GERENCIA")),
    "CARGO_ALTA_GERENCIA",
    "CARGO_ALTA_GERENCIA_SIN_IDENTIFICACION_VALIDA",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

df = add_obs(
    df,
    campo_ok("CARGO_ALTA_GERENCIA") &
    is_empty("NOMBRES_APELLIDOS_CARGO_GERENCIA"),
    "CARGO_ALTA_GERENCIA",
    "CARGO_ALTA_GERENCIA_SIN_NOMBRES_APELLIDOS",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

# REGLA: NOMBRES_APELLIDOS_CARGO_GERENCIA depende de campos 8 y 9
df = add_obs(
    df,
    campo_ok("NOMBRES_APELLIDOS_CARGO_GERENCIA") &
    (is_empty("IDENTIFICACION_CANT_INT_ALTA_GERENCIA") | 
     ~is_pos_number("IDENTIFICACION_CANT_INT_ALTA_GERENCIA")),
    "NOMBRES_APELLIDOS_CARGO_GERENCIA",
    "NOMBRES_APELLIDOS_SIN_IDENTIFICACION_VALIDA",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

df = add_obs(
    df,
    campo_ok("NOMBRES_APELLIDOS_CARGO_GERENCIA") &
    is_empty("CARGO_ALTA_GERENCIA"),
    "NOMBRES_APELLIDOS_CARGO_GERENCIA",
    "NOMBRES_APELLIDOS_SIN_CARGO",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

# REGLA: TIPO_DOC_ID y NRO_DOC_ID dependen de campo 10
df = add_obs(
    df,
    campo_ok("NOMBRES_APELLIDOS_CARGO_GERENCIA") & is_empty("TIPO_DOC_ID"),
    "TIPO_DOC_ID",
    "TIPO_DOC_ID_REQUERIDO_POR_NOMBRES_APELLIDOS",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

df = add_obs(
    df,
    campo_ok("NOMBRES_APELLIDOS_CARGO_GERENCIA") & is_empty("NRO_DOC_ID"),
    "NRO_DOC_ID",
    "NRO_DOC_ID_REQUERIDO_POR_NOMBRES_APELLIDOS",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

# REGLA: Campos de REP_LEGAL dependen de CODIGO_REP_LEGAL (campo 20)
campos_dep_rep_legal = [
    "NOMBRE_REP_LEGAL",
    "NACIONALIDAD_REP_LEGAL",
    "PAIS_RESIDENCIA_REP_LEGAL",
    "FECHA_NACIMIENTO_REP_LEGAL",
    "DOMICILIO_PARTICULAR_REP_LEGAL",
    "PROFESION_REP_LEGAL",
    "CODIGO_ACTIVIDAD_ECO_REP_LEGAL",
    "DESC_ACTIVIDAD_ECO_REP_LEGAL",
    "LUGAR_TRABAJO_REP_LEGAL",
    "CARGO_REP_LEGAL",
    "NIVEL_INGRESOS_REP_LEGAL",
    "NIVEL_INGRESOS_REP_LEGAL_SUS",
    "FECHA_INGRESO_TRAB_REP_LEGAL"
]

for c in campos_dep_rep_legal:
    df = add_obs(
        df,
        campo_ok("CODIGO_REP_LEGAL") & is_empty(c),
        c,
        f"{c}_REQUERIDO_POR_CODIGO_REP_LEGAL",
        TIPO_CONDICIONAL,
        SEV_ALTA
    )

# REGLA: NOMBRE_CONYUGE_REP_LEGAL depende de ESTADO_CIVIL_REP_LEGAL
df = add_obs(
    df,
    campo_ok("ESTADO_CIVIL_REP_LEGAL") &
    col("ESTADO_CIVIL_REP_LEGAL").rlike(r"(?i)(casad|concubin)") &
    is_empty("NOMBRE_CONYUGE_REP_LEGAL"),
    "NOMBRE_CONYUGE_REP_LEGAL",
    "NOMBRE_CONYUGE_REQUERIDO_POR_ESTADO_CIVIL",
    TIPO_CONDICIONAL,
    SEV_ALTA
)

# Validaciones de inconsistencias en condicionales
df = add_obs(
    df,
    campo_ok("DOMICILIO") & domicilio_invalido("DOMICILIO"),
    "DOMICILIO",
    "DOMICILIO_INVALIDO",
    TIPO_CONDICIONAL,
    SEV_MEDIA
)

df = add_obs(
    df,
    campo_ok("DOMICILIO_PARTICULAR_REP_LEGAL") & domicilio_invalido("DOMICILIO_PARTICULAR_REP_LEGAL"),
    "DOMICILIO_PARTICULAR_REP_LEGAL",
    "DOMICILIO_PARTICULAR_INVALIDO",
    TIPO_CONDICIONAL,
    SEV_MEDIA
)

# ============================================================================
# VALIDACIÓN PASO 4: CAMPOS OPCIONALES CONDICIONALES
# ============================================================================

print("=== PASO 4: Validando CAMPOS OPCIONALES CONDICIONALES ===")

# REGLA: Transacciones - campos 1, 2, 3 se requieren mutuamente
df = add_obs(
    df,
    (campo_ok("TIPO_CUENTA_ULTIMA_TRANSACCION") | campo_ok("CANAL_ATENCION_ULT_TRANS")) &
    is_empty("FECHA_ULTIMA_TRANSACCION"),
    "FECHA_ULTIMA_TRANSACCION",
    "FECHA_ULT_TRANS_REQUERIDA",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

df = add_obs(
    df,
    (campo_ok("FECHA_ULTIMA_TRANSACCION") | campo_ok("CANAL_ATENCION_ULT_TRANS")) &
    is_empty("TIPO_CUENTA_ULTIMA_TRANSACCION"),
    "TIPO_CUENTA_ULTIMA_TRANSACCION",
    "TIPO_CUENTA_ULT_TRANS_REQUERIDA",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

df = add_obs(
    df,
    (campo_ok("FECHA_ULTIMA_TRANSACCION") | campo_ok("TIPO_CUENTA_ULTIMA_TRANSACCION")) &
    is_empty("CANAL_ATENCION_ULT_TRANS"),
    "CANAL_ATENCION_ULT_TRANS",
    "CANAL_ULT_TRANS_REQUERIDO",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

# Validaciones específicas
df = add_obs(
    df,
    campo_ok("CORREO_ELECTRONICO_REP_LEGAL") & email_invalido("CORREO_ELECTRONICO_REP_LEGAL"),
    "CORREO_ELECTRONICO_REP_LEGAL",
    "EMAIL_FORMATO_INVALIDO",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

df = add_obs(
    df,
    campo_ok("TELEFONO_REP_LEGAL") & phone_invalido("TELEFONO_REP_LEGAL"),
    "TELEFONO_REP_LEGAL",
    "TELEFONO_INVALIDO",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

df = add_obs(
    df,
    campo_ok("TELEFONO") & phone_invalido("TELEFONO"),
    "TELEFONO",
    "TELEFONO_INVALIDO",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

# REGLA: CODIGO_REP_LEGAL depende de NOMBRE_REP_LEGAL
df = add_obs(
    df,
    campo_ok("NOMBRE_REP_LEGAL") & is_empty("CODIGO_REP_LEGAL"),
    "CODIGO_REP_LEGAL",
    "CODIGO_REP_LEGAL_REQUERIDO",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

# REGLA: Referencias - al menos una de 45, 46, 47 si tiene CODIGO_REP_LEGAL
df = add_obs(
    df,
    campo_ok("CODIGO_REP_LEGAL") &
    is_empty("REFERENCIAS_PERS_REP_LEGAL") &
    is_empty("REFERENCIAS_COM_REP_LEGAL") &
    is_empty("REFERENCIAS_BANCARIAS_REP_LEGAL"),
    "REFERENCIAS_PERS_REP_LEGAL",
    "AL_MENOS_UNA_REFERENCIA_REQUERIDA",
    TIPO_OPC_CONDICIONAL,
    SEV_MEDIA
)

# REGLA: CANTIDAD_ACTUALIZACIONES > 1 si FECHA_ULTIMA_ACTUALIZACION existe
df = add_obs(
    df,
    campo_ok("FECHA_ULTIMA_ACTUALIZACION") &
    (is_empty("CANTIDAD_ACTUALIZACIONES") | 
     (col("CANTIDAD_ACTUALIZACIONES").cast("int") <= 1)),
    "CANTIDAD_ACTUALIZACIONES",
    "CANT_ACTUALIZACIONES_DEBE_SER_MAYOR_1",
    TIPO_OPC_CONDICIONAL,
    SEV_BAJA
)

# ============================================================================
# VALIDACIÓN PASO 5: CAMPOS OPCIONALES
# ============================================================================

print("=== PASO 5: Validando CAMPOS OPCIONALES ===")

# Para opcionales solo validamos inconsistencias si tienen datos
for c in CAMP_OPCIONALES:
    if c in ALL_FIELDS:
        df = add_obs(
            df,
            campo_ok(c) & (text_invalido(c, 3) | tiene_palabra_invalida(c)),
            c,
            f"{c}_TEXTO_INVALIDO",
            TIPO_OPCIONAL,
            SEV_BAJA
        )

# ============================================================================
# GENERACIÓN DE CAMPOS RESUMEN PARA AUDITORÍA
# ============================================================================

print("=== Generando campos resumen ===")

df_final = (
    df
    # Contadores por tipo
    .withColumn("CANT_NULL", 
        size(expr("filter(OBS_ARRAY, x -> x.tipo = 'NULL')")))
    .withColumn("CANT_VACIO", 
        size(expr("filter(OBS_ARRAY, x -> x.tipo = 'VACIO')")))
    .withColumn("CANT_OBLIGADOS_FALTANTES", 
        size(expr(f"filter(OBS_ARRAY, x -> x.tipo = '{TIPO_OBLIGADO}')")))
    .withColumn("CANT_CONDICIONALES_FALTANTES", 
        size(expr(f"filter(OBS_ARRAY, x -> x.tipo = '{TIPO_CONDICIONAL}')")))
    .withColumn("CANT_OPC_CONDICIONALES_FALTANTES", 
        size(expr(f"filter(OBS_ARRAY, x -> x.tipo = '{TIPO_OPC_CONDICIONAL}')")))
    .withColumn("CANT_INCONSISTENCIAS_DATOS",
        size(expr(f"filter(OBS_ARRAY, x -> x.tipo = '{TIPO_OPCIONAL}')")))
    .withColumn("CANT_TOTAL_OBS", 
        size(col("OBS_ARRAY")))
    
    # Listados compactos para filtros rápidos
    .withColumn("OBS_CAMPOS_NULL_VACIO",
        expr("""
            concat_ws(', ', 
                transform(
                    filter(OBS_ARRAY, x -> x.tipo IN ('NULL', 'VACIO')),
                    x -> concat(x.campo, '_', lower(x.tipo))
                )
            )
        """))
    .withColumn("OBS_OBLIGADOS",
        expr(f"""
            concat_ws(', ',
                transform(
                    filter(OBS_ARRAY, x -> x.tipo = '{TIPO_OBLIGADO}'),
                    x -> x.motivo
                )
            )
        """))
    .withColumn("OBS_CONDICIONALES",
        expr(f"""
            concat_ws(', ',
                transform(
                    filter(OBS_ARRAY, x -> x.tipo = '{TIPO_CONDICIONAL}'),
                    x -> x.motivo
                )
            )
        """))
    .withColumn("OBS_INCONSISTENCIAS",
        expr(f"""
            concat_ws(', ',
                transform(
                    filter(OBS_ARRAY, x -> x.tipo IN ('{TIPO_OPC_CONDICIONAL}', '{TIPO_OPCIONAL}')),
                    x -> x.motivo
                )
            )
        """))
)

# ============================================================================
# DATAFRAMES FINALES PARA AUDITORÍA
# ============================================================================

# DataFrame de resumen ejecutivo
df_resumen_ejecutivo = (
    df_final
    .select(
        "CODIGO",
        "RAZON_SOCIAL",
        "NIT",
        "CANT_NULL",
        "CANT_VACIO",
        "CANT_OBLIGADOS_FALTANTES",
        "CANT_CONDICIONALES_FALTANTES",
        "CANT_OPC_CONDICIONALES_FALTANTES",
        "CANT_INCONSISTENCIAS_DATOS",
        "CANT_TOTAL_OBS",
        "OBS_CAMPOS_NULL_VACIO",
        "OBS_OBLIGADOS",
        "OBS_CONDICIONALES",
        "OBS_INCONSISTENCIAS"
    )
)

# DataFrame de detalle completo (para drill-down)
df_detalle_observaciones = (
    df_final
    .select("CODIGO", "RAZON_SOCIAL", "NIT", explode("OBS_ARRAY").alias("obs"))
    .select(
        "CODIGO",
        "RAZON_SOCIAL",
        "NIT",
        col("obs.idx").alias("IDX_CAMPO"),
        col("obs.campo").alias("CAMPO"),
        col("obs.tipo").alias("TIPO_VALIDACION"),
        col("obs.severidad").alias("SEVERIDAD"),
        col("obs.motivo").alias("DESCRIPCION")
    )
    .orderBy("CODIGO", "IDX_CAMPO")
)

# ============================================================================
# VISTAS Y FILTROS PARA AUDITORÍA
# ============================================================================

# Vista 1: Registros con problemas críticos (obligatorios)
df_criticos = df_resumen_ejecutivo.filter(
    (col("CANT_OBLIGADOS_FALTANTES") > 0) |
    (col("CANT_NULL") > 5)  # Más de 5 campos nulos
)

# Vista 2: Registros con problemas de dependencias
df_dependencias = df_resumen_ejecutivo.filter(
    (col("CANT_CONDICIONALES_FALTANTES") > 0)
)

# Vista 3: Registros con inconsistencias de datos
df_inconsistencias = df_resumen_ejecutivo.filter(
    (col("CANT_INCONSISTENCIAS_DATOS") > 0)
)

# Vista 4: Registros limpios (sin observaciones)
df_limpios = df_resumen_ejecutivo.filter(
    col("CANT_TOTAL_OBS") == 0
)

# ============================================================================
# ESTADÍSTICAS GENERALES
# ============================================================================

print("\n" + "="*80)
print("ESTADÍSTICAS GENERALES DE AUDITORÍA")
print("="*80)

total_registros = df_resumen_ejecutivo.count()
registros_con_obs = df_resumen_ejecutivo.filter(col("CANT_TOTAL_OBS") > 0).count()
registros_limpios = df_limpios.count()

print(f"\nTotal de registros: {total_registros:,}")
print(f"Registros con observaciones: {registros_con_obs:,} ({registros_con_obs/total_registros*100:.2f}%)")
print(f"Registros limpios: {registros_limpios:,} ({registros_limpios/total_registros*100:.2f}%)")

print("\n" + "-"*80)
print("DISTRIBUCIÓN DE OBSERVACIONES POR TIPO")
print("-"*80)

stats_por_tipo = df_resumen_ejecutivo.agg(
    sum("CANT_NULL").alias("TOTAL_NULL"),
    sum("CANT_VACIO").alias("TOTAL_VACIO"),
    sum("CANT_OBLIGADOS_FALTANTES").alias("TOTAL_OBLIGADOS"),
    sum("CANT_CONDICIONALES_FALTANTES").alias("TOTAL_CONDICIONALES"),
    sum("CANT_OPC_CONDICIONALES_FALTANTES").alias("TOTAL_OPC_CONDICIONALES"),
    sum("CANT_INCONSISTENCIAS_DATOS").alias("TOTAL_INCONSISTENCIAS"),
    sum("CANT_TOTAL_OBS").alias("TOTAL_OBSERVACIONES")
).collect()[0]

print(f"NULL: {stats_por_tipo['TOTAL_NULL']:,}")
print(f"VACÍO: {stats_por_tipo['TOTAL_VACIO']:,}")
print(f"Campos obligatorios faltantes: {stats_por_tipo['TOTAL_OBLIGADOS']:,}")
print(f"Campos condicionales faltantes: {stats_por_tipo['TOTAL_CONDICIONALES']:,}")
print(f"Campos opc. condicionales faltantes: {stats_por_tipo['TOTAL_OPC_CONDICIONALES']:,}")
print(f"Inconsistencias de datos: {stats_por_tipo['TOTAL_INCONSISTENCIAS']:,}")
print(f"TOTAL OBSERVACIONES: {stats_por_tipo['TOTAL_OBSERVACIONES']:,}")

print("\n" + "-"*80)
print("TOP 10 CAMPOS CON MÁS PROBLEMAS")
print("-"*80)

df_detalle_observaciones.groupBy("CAMPO").count() \
    .orderBy(desc("count")) \
    .limit(10) \
    .show(truncate=False)

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

print("\n" + "="*80)
print("GUARDANDO RESULTADOS")
print("="*80)

output_path = f"/user/T45109/AD/VerfCamposPJ/resultados_{fech_par}"

# Guardar resumen ejecutivo
df_resumen_ejecutivo.write.mode("overwrite") \
    .parquet(f"{output_path}/resumen_ejecutivo")
print(f"✓ Resumen ejecutivo guardado en: {output_path}/resumen_ejecutivo")

# Guardar detalle de observaciones
df_detalle_observaciones.write.mode("overwrite") \
    .parquet(f"{output_path}/detalle_observaciones")
print(f"✓ Detalle observaciones guardado en: {output_path}/detalle_observaciones")

# Guardar vistas específicas
df_criticos.write.mode("overwrite") \
    .parquet(f"{output_path}/registros_criticos")
print(f"✓ Registros críticos guardados en: {output_path}/registros_criticos")

df_dependencias.write.mode("overwrite") \
    .parquet(f"{output_path}/registros_con_dependencias")
print(f"✓ Registros con dependencias guardados en: {output_path}/registros_con_dependencias")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)

# ============================================================================
# EJEMPLOS DE USO PARA AUDITORÍA
# ============================================================================

print("\n\n" + "="*80)
print("EJEMPLOS DE CONSULTAS PARA AUDITORÍA")
print("="*80)

print("""
# Ver registros con campos obligatorios faltantes:
df_resumen_ejecutivo.filter(col("CANT_OBLIGADOS_FALTANTES") > 0).show(5, False)

# Ver detalle de un registro específico:
df_detalle_observaciones.filter(col("CODIGO") == "TU_CODIGO").show(100, False)

# Campos más problemáticos:
df_detalle_observaciones.groupBy("CAMPO", "TIPO_VALIDACION") \\
    .count().orderBy(desc("count")).show(20, False)

# Registros con NIT inválido:
df_resumen_ejecutivo.filter(col("OBS_INCONSISTENCIAS").contains("NIT_FORMATO_INVALIDO")).show(10, False)

# Registros con más de 10 observaciones:
df_resumen_ejecutivo.filter(col("CANT_TOTAL_OBS") > 10).orderBy(desc("CANT_TOTAL_OBS")).show(10, False)
""")