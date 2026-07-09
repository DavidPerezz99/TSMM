param(
    [string]$NvidiaInstallerPath = "",
    [switch]$RunCleanInstall,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Output ""
    Write-Output ("=== {0} ===" -f $Title)
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-NvidiaDisplayDrivers {
    Get-WmiObject Win32_PnPSignedDriver |
        Where-Object { $_.DeviceClass -eq "DISPLAY" -and $_.Manufacturer -match "NVIDIA" } |
        Select-Object DeviceName, DriverVersion, DriverDate, InfName, Manufacturer, DriverProviderName
}

function Get-NvidiaRelatedDrivers {
    Get-WmiObject Win32_PnPSignedDriver |
        Where-Object { $_.Manufacturer -match "NVIDIA" -or $_.DeviceName -match "NVIDIA" } |
        Select-Object DeviceName, DeviceClass, DriverVersion, DriverDate, InfName, Manufacturer
}

Write-Section "NVIDIA Driver Maintenance"
Write-Output ("Time: {0:u}" -f (Get-Date))
Write-Output ("DryRun: {0}" -f [bool]$DryRun)
Write-Output ("RunCleanInstall: {0}" -f [bool]$RunCleanInstall)

$displayDrivers = @(Get-NvidiaDisplayDrivers)
$relatedDrivers = @(Get-NvidiaRelatedDrivers)

Write-Section "Current NVIDIA Display Driver"
if ($displayDrivers.Count -eq 0) {
    Write-Output "No NVIDIA display driver was detected."
} else {
    $displayDrivers | Format-Table -AutoSize | Out-String | Write-Output
}

Write-Section "Installed NVIDIA Device Drivers"
if ($relatedDrivers.Count -eq 0) {
    Write-Output "No NVIDIA-related signed drivers were detected."
} else {
    $relatedDrivers | Sort-Object DeviceClass, DeviceName | Format-Table -AutoSize | Out-String | Write-Output
}

$displayInfNames = @($displayDrivers | Select-Object -ExpandProperty InfName -Unique)
Write-Section "Rollback Feasibility"
if ($displayInfNames.Count -le 1) {
    Write-Output "Local rollback is not available from driver store (only one display INF is present)."
    if ($displayInfNames.Count -eq 1) {
        Write-Output ("Active display INF: {0}" -f $displayInfNames[0])
    }
    Write-Output "Use clean reinstall with an NVIDIA installer package instead."
} else {
    Write-Output ("Multiple display INF packages detected: {0}" -f ($displayInfNames -join ", "))
}

if (-not $RunCleanInstall) {
    Write-Section "Next Step"
    Write-Output "To execute clean reinstall, rerun with:"
    Write-Output "  powershell -ExecutionPolicy Bypass -File scripts/windows_nvidia_driver_maintenance.ps1 -RunCleanInstall -NvidiaInstallerPath <path_to_installer.exe>"
    exit 0
}

if (-not (Test-IsAdmin)) {
    throw "Administrator privileges are required for clean reinstall. Open an elevated PowerShell window and run again."
}

if ([string]::IsNullOrWhiteSpace($NvidiaInstallerPath)) {
    throw "RunCleanInstall requires -NvidiaInstallerPath."
}

$installer = Get-Item -LiteralPath $NvidiaInstallerPath -ErrorAction Stop
Write-Section "Installer"
Write-Output ("Using installer: {0}" -f $installer.FullName)

$installerArgs = "-s -clean -noreboot"
Write-Output ("Installer arguments: {0}" -f $installerArgs)

if ($DryRun) {
    Write-Output "Dry run only. Installer was not executed."
    exit 0
}

Write-Section "Executing Clean Install"
$proc = Start-Process -FilePath $installer.FullName -ArgumentList $installerArgs -PassThru -Wait -WindowStyle Hidden
Write-Output ("Installer exit code: {0}" -f $proc.ExitCode)

if ($proc.ExitCode -ne 0) {
    throw ("NVIDIA installer failed with exit code {0}." -f $proc.ExitCode)
}

Write-Output "Clean install command completed. Reboot Windows now to finalize the driver stack."
exit 0
