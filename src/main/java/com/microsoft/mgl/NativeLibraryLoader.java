package com.microsoft.mgl;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

/**
 * Loads native libraries from the classpath, usually from a jar file.
 */
public final class NativeLibraryLoader {
    private static final Logger logger = LogManager.getLogger(NativeLibraryLoader.class);
    private static final String USE_LIBRARY_PATH = "USE_LIBRARY_PATH";
    private static final Set<String> loadedLibraries = new HashSet<String>();

    /**
     * Tries to load the native library from the classpath, usually from a jar file. <p>
     *
     * If the USE_LIBRARY_PATH environment variable is defined, the native library will be loaded from the
     * java.library.path instead of the classpath.
     *
     * @param tempDir  directory where the native library is extracted or null to use the system temp directory
     * @param libraries  name of the shared library without system dependent modifications, followed by the dependencies
     * @return true if the library was loaded successfully, false otherwise
     */
    public static synchronized boolean load(Path tempDir, String libraries) {

        String[] libraryNames = libraries.split(";");

        if (loadedLibraries.contains(libraryNames[0])) {
            return true;
        }

        for (int i = libraryNames.length; i-- > 0; ) {
            final String systemLibraryName = System.mapLibraryName(libraryNames[i]);

            // load from the java.library.path
            if (System.getenv(USE_LIBRARY_PATH) != null) {
                final String javaLibraryPath = System.getProperty("java.library.path");
                try {
                    logger.warn(String.format("OVERRIDE DEFAULT: Loading %s from %s", systemLibraryName, javaLibraryPath));
                    System.loadLibrary(libraryNames[i]);
                } catch (Exception|Error e) {
                    logger.warn(String.format("OVERRIDE DEFAULT: Unable to load %s from %s", systemLibraryName, javaLibraryPath));
                    return false;
                }
            }
            else {
                if(tempDir == null) {
                    try {
                        tempDir = Files.createTempDirectory(null);
                    }
                    catch (IOException | Error e) {
                        logger.warn(String.format("Unable to create temporary directory (%s)", e.getMessage()));
                        return false;
                    }
                }

                // load from the java classpath
                final String resourcePath = "native/" + systemLibraryName;
                final URL inputUrl = NativeLibraryLoader.class.getResource(resourcePath);
                if (inputUrl == null) {
                    logger.warn("Unable to find native library: " + resourcePath);
                    return false;
                }
                logger.info(String.format("Loading %s from %s", systemLibraryName, inputUrl.toString()));

                try {
                    final Path tempPath = Files.createFile(tempDir.resolve(systemLibraryName));
                    final File temp = tempPath.toFile();
                    temp.deleteOnExit();
                    FileUtils.copyURLToFile(inputUrl, temp);
                    logger.debug(String.format("Extracting %s to %s", systemLibraryName, temp.getAbsolutePath()));
                    System.load(temp.getAbsolutePath());
                    loadedLibraries.add(libraryNames[i]);
                } catch (Exception | Error e) {
                    logger.warn(String.format("Unable to load %s from %s (%s)", systemLibraryName, resourcePath, e.getMessage()));
                    return false;
                }
            }
        }

        return true;
    }
}
