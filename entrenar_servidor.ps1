<#
Entrena modelos en este PC usando los datos de la base de datos del SERVIDOR.

  .\entrenar_servidor.ps1                       # entrenamiento completo
  .\entrenar_servidor.ps1 -Publicar             # entrena y copia el modelo al servidor
  .\entrenar_servidor.ps1 --metrica recall      # cualquier opción de src.entrenar se pasa tal cual

Cómo funciona:
  1. Abre un túnel SSH: localhost:5434 (PC) -> 127.0.0.1:5432 (PostgreSQL del servidor).
     La base de datos nunca queda expuesta a internet.
  2. Lee la contraseña de la base desde el .env del servidor (no se guarda en el PC).
  3. Ejecuta python -m src.entrenar con DATABASE_URL apuntando al túnel.
  4. Cierra el túnel. Con -Publicar, copia el modelo nuevo al servidor.
#>
param([switch]$Publicar, [Parameter(ValueFromRemainingArguments = $true)] $ArgsEntrenar)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
. .\servidor.ps1

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "No existe ml\.venv. Créelo primero (ver ml\README.md)." }

Write-Host "1/4 Leyendo la configuración de la base de datos del servidor..." -ForegroundColor Cyan
$clave = (& ssh @OpcionesSsh $Destino "grep '^POSTGRES_PASSWORD=' $CarpetaApp/.env | cut -d= -f2-").Trim()
$usuarioBd = (& ssh @OpcionesSsh $Destino "grep '^POSTGRES_USER=' $CarpetaApp/.env | cut -d= -f2-").Trim()
if (-not $clave) { throw "No se pudo leer POSTGRES_PASSWORD del servidor. ¿Está encendida la máquina virtual?" }
if (-not $usuarioBd) { $usuarioBd = 'postgres' }

Write-Host "2/4 Abriendo túnel SSH (localhost:$PuertoLocal -> servidor)..." -ForegroundColor Cyan
$tunel = Start-Process ssh -PassThru -WindowStyle Hidden -ArgumentList (
    $OpcionesSsh + @('-N', '-o', 'ExitOnForwardFailure=yes', '-L', "${PuertoLocal}:127.0.0.1:5432", $Destino))
$listo = $false
foreach ($i in 1..20) {
    Start-Sleep -Milliseconds 500
    if ($tunel.HasExited) { break }
    try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', $PuertoLocal); $c.Close(); $listo = $true; break } catch {}
}
if (-not $listo) { if (-not $tunel.HasExited) { $tunel.Kill() }; throw "No se pudo abrir el túnel (¿puerto $PuertoLocal ocupado o servidor apagado?)." }

try {
    $env:DATABASE_URL = "postgresql+psycopg://${usuarioBd}:${clave}@localhost:${PuertoLocal}/desercion_db"
    Write-Host "3/4 Entrenando con los datos del servidor..." -ForegroundColor Cyan
    & $python -m src.entrenar @ArgsEntrenar
    if ($LASTEXITCODE -ne 0) { throw "El entrenamiento terminó con error ($LASTEXITCODE)." }
}
finally {
    Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue
    if (-not $tunel.HasExited) { $tunel.Kill() }
    Write-Host "    Túnel cerrado." -ForegroundColor DarkGray
}

if ($Publicar) {
    Write-Host "4/4 Publicando el modelo en el servidor..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'publicar_modelo.ps1')
} else {
    Write-Host "4/4 Listo. Para enviarlo al servidor ejecute: .\publicar_modelo.ps1" -ForegroundColor Green
}
