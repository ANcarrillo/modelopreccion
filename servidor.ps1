# Configuración compartida por entrenar_servidor.ps1 y publicar_modelo.ps1.
# Cambie estos valores si la IP del servidor o la ubicación de la clave cambian.

$Servidor   = '20.15.200.218'                                  # IP pública de la máquina virtual
$Usuario    = 'azureuser'
$Clave      = Join-Path $env:USERPROFILE 'Downloads\predic_key.pem'  # clave SSH descargada de Azure
$CarpetaApp = '~/app'                                          # proyecto en el servidor
$PuertoLocal = 5434                                            # puerto en el PC para el túnel

if (-not (Test-Path $Clave)) {
    $alternativa = Join-Path $env:USERPROFILE '.ssh\predic_key.pem'
    if (Test-Path $alternativa) { $Clave = $alternativa }
    else { throw "No se encontró la clave SSH en '$Clave'. Edite ml\servidor.ps1." }
}

$Destino = "$Usuario@$Servidor"
$OpcionesSsh = @('-i', $Clave, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15')
