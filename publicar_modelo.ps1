<#
Copia un modelo entrenado (.pkl + .json) al servidor.

  .\publicar_modelo.ps1                                  # el más reciente de ml\models
  .\publicar_modelo.ps1 modelo_desercion_20261124_101500 # uno en particular (sin extensión)

Después, en la web: Modelo ML -> "Registrar y activar".
No hace falta reiniciar nada: el backend lee la carpeta de modelos directamente.
#>
param([string]$Nombre)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
. .\servidor.ps1

$carpeta = Join-Path $PSScriptRoot 'models'
if ($Nombre) {
    $pkl = Get-Item (Join-Path $carpeta "$Nombre.pkl")
} else {
    $pkl = Get-ChildItem $carpeta -Filter 'modelo_*.pkl' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
if (-not $pkl) { throw "No hay modelos en ml\models. Entrene uno primero." }
$json = [IO.Path]::ChangeExtension($pkl.FullName, '.json')
if (-not (Test-Path $json)) { throw "Falta el archivo de metadatos $([IO.Path]::GetFileName($json))." }

$meta = Get-Content $json -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "Modelo: $($pkl.BaseName)  ($($meta.algoritmo), F1 $($meta.metricas.f1), AUC $($meta.metricas.auc_roc), recall $($meta.metricas.recall))" -ForegroundColor Cyan

$existe = & ssh @OpcionesSsh $Destino "test -f $CarpetaApp/ml/models/$($pkl.Name) && echo si"
if ($existe -eq 'si') { Write-Host "Ese modelo ya está en el servidor; no se copia de nuevo." -ForegroundColor Yellow; return }

& scp @OpcionesSsh $pkl.FullName $json "${Destino}:$CarpetaApp/ml/models/"
if ($LASTEXITCODE -ne 0) { throw "No se pudo copiar el modelo al servidor." }

Write-Host "Copiado al servidor. Ahora en la web: Modelo ML -> 'Registrar y activar'." -ForegroundColor Green
Write-Host "https://$($Servidor -replace '\.', '-').sslip.io/modelo" -ForegroundColor Green
