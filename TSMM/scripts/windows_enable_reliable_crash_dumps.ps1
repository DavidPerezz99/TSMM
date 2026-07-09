param(
    [int]$PageFileMB = 16384,
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

function Get-RamMB {
    $cs = Get-CimInstance Win32_ComputerSystem
    return [int]([math]::Round(($cs.TotalPhysicalMemory / 1MB), 0))
}

function Ensure-FreeSpace {
    param([int]$RequiredMB)
    $drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
    if (-not $drive) {
        throw "Unable to read C: drive free space."
    }
    $freeMB = [int]([math]::Floor(($drive.FreeSpace / 1MB)))
    if ($freeMB -lt $RequiredMB) {
        throw ("Insufficient free space on C:. Required >= {0} MB, current = {1} MB" -f $RequiredMB, $freeMB)
    }
    Write-Output ("C: free space check passed ({0} MB free)." -f $freeMB)
}

function Set-CrashControlValue {
    param(
        [string]$Name,
        [object]$Value,
        [Microsoft.Win32.RegistryValueKind]$Kind
    )
    $path = "HKLM:\SYSTEM\CurrentControlSet\Control\CrashControl"
    New-ItemProperty -Path $path -Name $Name -PropertyType $Kind -Value $Value -Force | Out-Null
}

Write-Section "Crash Dump Hardening"
Write-Output ("Time: {0:u}" -f (Get-Date))
Write-Output ("DryRun: {0}" -f [bool]$DryRun)

$ramMB = Get-RamMB
$recommendedMB = [int]([math]::Max(16384, [math]::Min(32768, [math]::Ceiling($ramMB * 0.5))))
if ($PageFileMB -lt $recommendedMB) {
    Write-Output ("Requested pagefile size ({0} MB) is below recommended ({1} MB). Using recommended value." -f $PageFileMB, $recommendedMB)
    $PageFileMB = $recommendedMB
}

Write-Output ("Total RAM: {0} MB" -f $ramMB)
Write-Output ("Target fixed pagefile: {0} MB" -f $PageFileMB)

if (-not (Test-IsAdmin)) {
    if ($DryRun) {
        Write-Output "Dry run note: administrator privileges are required for real changes."
        exit 0
    }
    throw "Administrator privileges are required. Open an elevated PowerShell window and run this script again."
}

Ensure-FreeSpace -RequiredMB ($PageFileMB + 2048)

Write-Section "Planned CrashControl Values"
Write-Output "CrashDumpEnabled = 2 (Kernel memory dump)"
Write-Output "DumpFile = %SystemRoot%\\MEMORY.DMP"
Write-Output "MinidumpDir = %SystemRoot%\\Minidump"
Write-Output "MinidumpsCount = 20"
Write-Output "LogEvent = 1"
Write-Output "Overwrite = 1"
Write-Output "AutoReboot = 1"
Write-Output "AlwaysKeepMemoryDump = 1"
Write-Output "EnableLogFile = 1"

if ($DryRun) {
    Write-Output "Dry run only. No registry or pagefile changes were applied."
    exit 0
}

Write-Section "Applying CrashControl Settings"
Set-CrashControlValue -Name "CrashDumpEnabled" -Value 2 -Kind DWord
Set-CrashControlValue -Name "DumpFile" -Value "%SystemRoot%\\MEMORY.DMP" -Kind ExpandString
Set-CrashControlValue -Name "MinidumpDir" -Value "%SystemRoot%\\Minidump" -Kind ExpandString
Set-CrashControlValue -Name "MinidumpsCount" -Value 20 -Kind DWord
Set-CrashControlValue -Name "LogEvent" -Value 1 -Kind DWord
Set-CrashControlValue -Name "Overwrite" -Value 1 -Kind DWord
Set-CrashControlValue -Name "AutoReboot" -Value 1 -Kind DWord
Set-CrashControlValue -Name "AlwaysKeepMemoryDump" -Value 1 -Kind DWord
Set-CrashControlValue -Name "EnableLogFile" -Value 1 -Kind DWord

Write-Section "Applying Fixed Pagefile"
$cs = Get-WmiObject Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    $cs.AutomaticManagedPagefile = $false
    $null = $cs.Put()
    Write-Output "Disabled AutomaticManagedPagefile."
}

$mmPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
$pagingValue = "C:\pagefile.sys {0} {1}" -f $PageFileMB, $PageFileMB
Set-ItemProperty -Path $mmPath -Name "PagingFiles" -Value @($pagingValue)
Set-ItemProperty -Path $mmPath -Name "ExistingPageFiles" -Value @("\\??\\C:\pagefile.sys")
Set-ItemProperty -Path $mmPath -Name "TempPageFile" -Value 0 -Type DWord
Write-Output ("Configured PagingFiles registry value: {0}" -f $pagingValue)

Write-Section "Final Verification"
reg query "HKLM\SYSTEM\CurrentControlSet\Control\CrashControl"
reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v PagingFiles
Get-CimInstance Win32_ComputerSystem | Select-Object AutomaticManagedPagefile | Format-List

Write-Output "Crash dump hardening completed. Reboot Windows for pagefile changes to take full effect."
exit 0
