$training = Get-CimInstance Win32_Process -Filter "Name = 'bun.exe'" | Where-Object { $_.CommandLine -like '*train.ts*' }
foreach ($entry in $training) {
  $process = Get-Process -Id $entry.ProcessId
  $process.ProcessorAffinity = 255
  $process.PriorityClass = 'High'
  Write-Host "Pinned process $($entry.ProcessId) to logical processors 0-7"
}
