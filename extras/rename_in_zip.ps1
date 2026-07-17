Add-Type -AssemblyName System.IO.Compression.FileSystem

$zipPath = "e:\New folder\coding_arc\Font_Identifier_AI\ttf_files.zip"
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, 'Update')

$entriesToRename = @()
foreach ($entry in $zip.Entries) {
    if ($entry.FullName -match '#') {
        $entriesToRename += $entry
    }
}

$count = 0
foreach ($entry in $entriesToRename) {
    $newName = $entry.FullName.Replace('#', '_')
    Write-Host "Renaming $($entry.FullName) to $newName"
    
    # Create new entry
    $newEntry = $zip.CreateEntry($newName)
    
    # Copy data
    $oldStream = $entry.Open()
    $newStream = $newEntry.Open()
    $oldStream.CopyTo($newStream)
    
    $oldStream.Close()
    $newStream.Close()
    
    # Delete old entry
    $entry.Delete()
    $count++
}

$zip.Dispose()
Write-Host "Successfully renamed $count files instantly inside the ZIP!"
